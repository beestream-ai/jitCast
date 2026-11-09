/*
 * Copyright (c) 2026 Beestream Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitcast.tirex2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.host.SeriesWindows;
import ai.beestream.jitcast.host.SignFlip;

/**
 * {@code ForecastModel.forecast()} around any {@link ForwardPass}: windows of at most
 * {@code batchSize} series and {@code maxTokens} tokens (rows × patches; a series with covariates
 * has one row per target and covariate), batch assembly ({@code PostProcessor.transform_input} and the padding
 * of {@code TiRex2._predict}), the forward pass, slicing out the horizon, calibration
 * ({@code transform_output}) and the optional sign-flip test-time augmentation.
 *
 * <p>Contexts are always padded to the full model length ({@code pad_context=True}, the default).
 */
public final class ForecastPipeline {

    /** The default {@code batch_size} of {@code ForecastModel.forecast}. */
    public static final int DEFAULT_BATCH_SIZE = 512;

    /**
     * {@return the most tokens one forward pass takes: {@code -Dtirex.maxTokens}, default 131,072}
     * An engine sizes its buffers by tokens, so this bounds its memory whatever the series' covariate
     * counts (about 4 GB per plan for the TornadoVM engine). A series with more tokens runs alone.
     * Splitting a batch does not change the forecasts: a row's position in the model input does
     * not depend on the other series of its batch.
     */
    public static long defaultMaxTokens() {
        return ai.beestream.jitcast.util.Options.number("tirex.maxTokens", 131_072L, 1);
    }

    private final TiRex2Config config;
    private final ForwardPass forward;
    private final int batchSize;
    private final long maxTokens;
    private final boolean ttaDiff;
    private final boolean ttaSignFlip;

    /**
     * A pipeline with the checkpoint's test-time augmentation, the default batch size and the
     * default token budget.
     *
     * @param config the checkpoint's configuration
     * @param forward the forward pass to run
     */
    public ForecastPipeline(TiRex2Config config, ForwardPass forward) {
        this(config, forward, DEFAULT_BATCH_SIZE, config.ttaDiff(), config.ttaSignFlip());
    }

    /**
     * A pipeline with explicit batch size and test-time augmentation, and the default token budget.
     *
     * @param config the checkpoint's configuration
     * @param forward the forward pass to run
     * @param batchSize the most series per forward pass
     * @param ttaDiff whether to apply trend differencing
     * @param ttaSignFlip whether to average with the forecast of the negated series
     */
    public ForecastPipeline(TiRex2Config config, ForwardPass forward, int batchSize, boolean ttaDiff, boolean ttaSignFlip) {
        this(config, forward, batchSize, defaultMaxTokens(), ttaDiff, ttaSignFlip);
    }

    /**
     * A pipeline with explicit settings.
     *
     * @param config the checkpoint's configuration
     * @param forward the forward pass to run
     * @param batchSize the most series per forward pass
     * @param maxTokens the most tokens (rows × patches) per forward pass
     * @param ttaDiff whether to apply trend differencing
     * @param ttaSignFlip whether to average with the forecast of the negated series
     * @throws IllegalArgumentException if a size is below 1
     */
    public ForecastPipeline(TiRex2Config config, ForwardPass forward, int batchSize, long maxTokens, boolean ttaDiff,
            boolean ttaSignFlip) {
        if (batchSize < 1) {
            throw new IllegalArgumentException("batchSize must be >= 1");
        }
        if (maxTokens < 1) {
            throw new IllegalArgumentException("maxTokens must be >= 1");
        }
        this.config = config;
        this.forward = forward;
        this.batchSize = batchSize;
        this.maxTokens = maxTokens;
        this.ttaDiff = ttaDiff;
        this.ttaSignFlip = ttaSignFlip;
    }

    /** {@return whether trend differencing is on: the checkpoint's {@code tta_diff} unless overridden} */
    public boolean trendDiff() {
        return ttaDiff;
    }

    /** {@return whether sign-flip TTA is on: the checkpoint's {@code tta_sign_flip} unless overridden} */
    public boolean signFlip() {
        return ttaSignFlip;
    }

    /**
     * {@return the same pipeline with sign-flip TTA on or off, as the {@code tta_sign_flip} argument of predict}
     *
     * @param signFlip whether to average with the forecast of the negated series
     */
    public ForecastPipeline withSignFlip(boolean signFlip) {
        return new ForecastPipeline(config, forward, batchSize, maxTokens, ttaDiff, signFlip);
    }

    /**
     * {@return the same pipeline with another token budget per forward pass}
     *
     * @param tokens the most tokens per forward pass
     */
    public ForecastPipeline withMaxTokens(long tokens) {
        return new ForecastPipeline(config, forward, batchSize, tokens, ttaDiff, ttaSignFlip);
    }

    /**
     * Forecasts every series, as {@link ai.beestream.jitcast.api.ForecastEngine#forecast}.
     *
     * @param series the inputs
     * @param horizon the number of steps to forecast, at least 1
     * @return one forecast per series, in order
     * @throws IllegalArgumentException if {@code horizon} is below 1 or a series has inputs the
     *     model does not take
     */
    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        if (horizon < 1) {
            throw new IllegalArgumentException("horizon must be >= 1, got " + horizon);
        }
        for (TimeSeries ts : series) {
            if (ts.hasCategorical()) {
                throw new IllegalArgumentException("TiRex-2 does not take categorical covariates");
            }
        }
        // like TiRex2._predict_once, a horizon beyond futureLen is truncated to it
        int h = Math.min(horizon, config.futureLen());
        List<Forecast> out = new ArrayList<>(series.size());
        for (int[] w : SeriesWindows.split(series, batchSize, maxTokens, config.numPatches())) {
            out.addAll(predict(series.subList(w[0], w[1]), h));
        }
        return out;
    }

    /** {@code TiRex2.predict}: one pass, or the average of the pass and its sign-flipped mirror. */
    List<Forecast> predict(List<TimeSeries> series, int h) {
        List<float[][][]> a = predictOnce(series, h);
        if (!ttaSignFlip || series.isEmpty()) {
            return wrap(a);
        }
        List<float[][][]> b = predictOnce(series.stream().map(SignFlip::negate).toList(), h);
        int[] complement = SignFlip.complement(config.quantiles());
        for (int s = 0; s < a.size(); s++) {
            SignFlip.average(a.get(s), b.get(s), complement, h);
        }
        return wrap(a);
    }

    private List<Forecast> wrap(List<float[][][]> values) {
        return values.stream().map(v -> new Forecast(config.quantiles().clone(), v)).toList();
    }

    /** The rows, group ids and target flags of one assembled batch, plus what the output side needs. */
    record Assembled(ModelBatch batch, int[][] targetRows, boolean[][] differenced, float[][] lastValues) {}

    Assembled assemble(List<TimeSeries> series, int h) {
        int n = series.size();
        int maxLen = series.stream().mapToInt(TimeSeries::length).max().orElse(0);
        int modelLen = config.paddedLength();
        List<float[]> rows = new ArrayList<>();
        List<Float> groups = new ArrayList<>();
        List<Boolean> targets = new ArrayList<>();
        int[][] targetRows = new int[n][];
        boolean[][] differenced = new boolean[n][];
        float[][] last = new float[n][];

        for (int s = 0; s < n; s++) {
            TimeSeries ts = series.get(s);
            int T = ts.length();
            int leftPad = maxLen - T;
            float group = s + 1;
            boolean[] diff = new boolean[ts.targets()];
            last[s] = new float[ts.targets()];
            boolean anyDiff = false;
            for (int v = 0; v < ts.targets(); v++) {
                diff[v] = ttaDiff && PostProcessor.isTrending(ts.target()[v]);
                anyDiff |= diff[v];
                last[s][v] = PostProcessor.lastValue(ts.target()[v]);
            }
            differenced[s] = diff;
            targetRows[s] = new int[ts.targets()];
            for (int v = 0; v < ts.targets(); v++) {
                float[] values = diff[v] && anyDiff ? PostProcessor.diff(ts.target()[v]) : ts.target()[v];
                targetRows[s][v] = rows.size();
                rows.add(place(values, T, leftPad, h, maxLen, modelLen));
                groups.add(group);
                targets.add(true);
            }
            for (int v = 0; v < ts.pastCount(); v++) {
                rows.add(place(ts.pastCovariates()[v], T, leftPad, h, maxLen, modelLen));
                groups.add(group);
                targets.add(false);
            }
            for (int v = 0; v < ts.futureCount(); v++) {
                float[] fc = ts.futureCovariates()[v];
                if (fc.length < T + h) {
                    throw new IllegalArgumentException("future covariates of series " + s + " need at least "
                            + (T + h) + " steps, got " + fc.length);
                }
                rows.add(place(fc, T + h, leftPad, 0, maxLen, modelLen));
                groups.add(group);
                targets.add(false);
            }
        }
        float[] g = new float[rows.size()];
        boolean[] t = new boolean[rows.size()];
        for (int i = 0; i < g.length; i++) {
            g[i] = groups.get(i);
            t[i] = targets.get(i);
        }
        return new Assembled(new ModelBatch(rows.toArray(float[][]::new), g, t), targetRows, differenced, last);
    }

    /**
     * Lays out the first {@code count} values of a row the way transform_input and _predict do:
     * left-padded to the batch's longest context, right-padded over the horizon ({@code rightPad})
     * and then to {@code futureLen}, then left-padded (or cut from the left) to the model length.
     */
    private float[] place(float[] values, int count, int leftPad, int rightPad, int maxLen, int modelLen) {
        // transform_input: [leftPad NaN][values][rightPad NaN] = maxLen + h; _predict adds futureLen - h
        int len = maxLen + config.futureLen();
        float[] row = new float[modelLen];
        Arrays.fill(row, Float.NaN);
        int offset = modelLen - len; // left pad (> 0) or cut from the left (< 0)
        for (int i = 0; i < count; i++) {
            int pos = offset + leftPad + i;
            if (pos >= 0) {
                row[pos] = values[i];
            }
        }
        return row;
    }

    /** One pass: assemble, forward, slice the horizon, calibrate each target row. */
    List<float[][][]> predictOnce(List<TimeSeries> series, int h) {
        Assembled a = assemble(series, h);
        float[][][] out = forward.forward(a.batch());
        // pred[..., :-P][..., -futureLen:][..., :h]
        int from = config.paddedLength() - config.patchSize() - config.futureLen();
        List<float[][][]> result = new ArrayList<>(series.size());
        for (int s = 0; s < series.size(); s++) {
            int vt = a.targetRows()[s].length;
            float[][][] f = new float[vt][][];
            for (int v = 0; v < vt; v++) {
                float[][] rowOut = out[a.targetRows()[s][v]];
                float[][] q = new float[rowOut.length][];
                for (int k = 0; k < q.length; k++) {
                    q[k] = Arrays.copyOfRange(rowOut[k], from, from + h);
                }
                PostProcessor.calibrate(q, a.differenced()[s][v], a.lastValues()[s][v]);
                f[v] = q;
            }
            result.add(f);
        }
        return result;
    }
}
