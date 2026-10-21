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
package ai.beestream.jitcast.chronos2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;

/**
 * {@code Chronos2Pipeline.predict()} around any {@link ModelForward}: batches of consecutive series
 * up to {@code batchSize} rows (one row per target, past-only covariate and known covariate, each
 * cut to the context length and left-padded with {@code NaN}), one group per series (or one group
 * per batch with cross-learning), a first forward pass of up to {@code maxOutputPatches} patches,
 * and past it the long-horizon unrolling: every row is repeated once per unrolled quantile level,
 * each copy's context is extended with that level of the forecast so far (the known covariates with
 * their real future), and the next pass's forecasts of all copies are merged into the quantile
 * levels again as a weighted sample.
 *
 * <p>Within a batch the series do not affect each other's forecasts unless cross-learning is on.
 * A forward pass larger than {@code maxTokens} tokens runs as several, split between groups, which
 * does not change the forecasts. Categorical covariates are not supported.
 */
public final class Chronos2Pipeline {

    /** The reference's default batch size, in rows. */
    public static final int DEFAULT_BATCH_SIZE = 256;

    /** The most tokens one forward pass takes ({@code -Dchronos2.maxTokens}, default 131,072). */
    public static final long DEFAULT_MAX_TOKENS = Long.getLong("chronos2.maxTokens", 131_072L);

    /** The reference's default levels for the long-horizon unrolling. */
    private static final float[] UNROLLED = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};

    /**
     * The forecaster's settings, with the reference's defaults.
     *
     * @param contextLength context cap in time steps (the most recent are kept), at most the model's
     * @param crossLearning one group per batch: every series of a batch attends every other
     * @param maxOutputPatches patches per forward pass before the long-horizon unrolling
     * @param unrolledQuantiles the levels the long-horizon unrolling feeds back, a subset of the
     *     model's
     */
    public record Options(int contextLength, boolean crossLearning, int maxOutputPatches, float[] unrolledQuantiles) {

        public static Options defaults(Chronos2Config c) {
            return new Options(c.contextLength(), false, c.maxOutputPatches(), UNROLLED.clone());
        }

        public Options withCrossLearning(boolean on) {
            return new Options(contextLength, on, maxOutputPatches, unrolledQuantiles);
        }

        public Options withContextLength(int steps) {
            return new Options(steps, crossLearning, maxOutputPatches, unrolledQuantiles);
        }
    }

    private final Chronos2Config config;
    private final ModelForward forward;
    private final Options options;
    private final int batchSize;
    private final long maxTokens;

    public Chronos2Pipeline(Chronos2Config config, ModelForward forward) {
        this(config, forward, Options.defaults(config), DEFAULT_BATCH_SIZE, DEFAULT_MAX_TOKENS);
    }

    public Chronos2Pipeline(Chronos2Config config, ModelForward forward, Options options, int batchSize, long maxTokens) {
        if (batchSize < 1 || maxTokens < 1) {
            throw new IllegalArgumentException("batchSize and maxTokens must be >= 1");
        }
        if (options.contextLength() < 1 || options.maxOutputPatches() < 1) {
            throw new IllegalArgumentException("contextLength and maxOutputPatches must be >= 1");
        }
        float[] levels = config.quantileLevels();
        for (float u : options.unrolledQuantiles()) {
            boolean found = false;
            for (float l : levels) {
                found |= l == u;
            }
            if (!found) {
                throw new IllegalArgumentException("unrolled quantile " + u + " is not one of the model's levels");
            }
        }
        this.config = config;
        this.forward = forward;
        this.options = options;
        this.batchSize = batchSize;
        this.maxTokens = maxTokens;
    }

    public Options options() {
        return options;
    }

    public Chronos2Pipeline withOptions(Options o) {
        return new Chronos2Pipeline(config, forward, o, batchSize, maxTokens);
    }

    public Chronos2Pipeline withBatchSize(int rows) {
        return new Chronos2Pipeline(config, forward, options, rows, maxTokens);
    }

    public Chronos2Pipeline withMaxTokens(long tokens) {
        return new Chronos2Pipeline(config, forward, options, batchSize, tokens);
    }

    /** The rows of one series: targets, past-only covariates, then known covariates. */
    private record Prepared(float[][] context, float[][] future, int targets) {}

    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        if (horizon < 1) {
            throw new IllegalArgumentException("horizon must be >= 1, got " + horizon);
        }
        int ctx = Math.min(options.contextLength(), config.contextLength());
        List<Prepared> prepared = new ArrayList<>(series.size());
        for (TimeSeries ts : series) {
            if (ts.hasCategorical()) {
                throw new IllegalArgumentException("Chronos-2 does not support categorical covariates; encode them as numbers");
            }
            prepared.add(prepare(ts, horizon, ctx));
        }
        List<Forecast> out = new ArrayList<>(series.size());
        // Chronos2Dataset's sequential batches: series are added while the batch has fewer rows than batchSize
        for (int start = 0; start < prepared.size();) {
            int end = start, rows = 0;
            while (end < prepared.size() && rows < batchSize) {
                rows += prepared.get(end).context().length;
                end++;
            }
            out.addAll(predictBatch(prepared.subList(start, end), horizon));
            start = end;
        }
        return out;
    }

    private static Prepared prepare(TimeSeries ts, int horizon, int ctx) {
        int len = ts.length();
        int rows = ts.targets() + ts.pastCount() + ts.futureCount();
        int keep = Math.min(len, ctx);
        float[][] context = new float[rows][];
        float[][] future = new float[rows][horizon];
        int r = 0;
        for (float[] row : ts.target()) {
            context[r++] = Arrays.copyOfRange(row, len - keep, len);
        }
        for (int v = 0; v < ts.pastCount(); v++) {
            context[r++] = Arrays.copyOfRange(ts.pastCovariates()[v], len - keep, len);
        }
        for (int v = 0; v < ts.futureCount(); v++) {
            float[] fc = ts.futureCovariates()[v];
            if (fc.length < len + horizon) {
                throw new IllegalArgumentException("future covariates need at least " + (len + horizon) + " steps, got "
                        + fc.length);
            }
            context[r] = Arrays.copyOfRange(fc, len - keep, len);
            System.arraycopy(fc, len, future[r], 0, horizon);
            r++;
        }
        for (int i = 0; i < ts.targets() + ts.pastCount(); i++) {
            Arrays.fill(future[i], Float.NaN);
        }
        return new Prepared(context, future, ts.targets());
    }

    /** {@code _predict_batch}: one batch's forecasts, per series {@code [targets][quantiles][horizon]}. */
    private List<Forecast> predictBatch(List<Prepared> batch, int horizon) {
        int len = batch.stream().mapToInt(p -> p.context()[0].length).max().orElse(0);
        List<float[]> ctxRows = new ArrayList<>();
        List<float[]> futRows = new ArrayList<>();
        List<Integer> groups = new ArrayList<>();
        int[] firstRow = new int[batch.size()];
        for (int s = 0; s < batch.size(); s++) {
            Prepared p = batch.get(s);
            firstRow[s] = ctxRows.size();
            for (int i = 0; i < p.context().length; i++) {
                float[] row = new float[len];
                int l = p.context()[i].length;
                Arrays.fill(row, 0, len - l, Float.NaN);
                System.arraycopy(p.context()[i], 0, row, len - l, l);
                ctxRows.add(row);
                futRows.add(p.future()[i]);
                groups.add(options.crossLearning() ? 0 : s);
            }
        }
        float[][] context = ctxRows.toArray(float[][]::new);
        float[][] future = futRows.toArray(float[][]::new);
        int[] group = groups.stream().mapToInt(Integer::intValue).toArray();
        int rows = context.length, q = config.numQuantiles();

        float[][][] prediction = predictStep(context, group, future, outputPatches(horizon));
        List<float[][][]> predictions = new ArrayList<>();
        predictions.add(prediction);
        int remaining = horizon - prediction[0][0].length;
        if (remaining > 0) {
            float[] unrolled = options.unrolledQuantiles();
            int paths = unrolled.length;
            // (batch, path) rows; a group mixes only the rows of the same path
            float[][] c2 = new float[rows * paths][];
            float[][] f2 = new float[rows * paths][];
            int[] g2 = new int[rows * paths];
            for (int b = 0; b < rows; b++) {
                for (int n = 0; n < paths; n++) {
                    c2[b * paths + n] = context[b];
                    f2[b * paths + n] = future[b];
                    g2[b * paths + n] = group[b] * paths + n;
                }
            }
            context = c2;
            future = f2;
            group = g2;
            float[] pathMass = QuantileMath.probabilityMass(unrolled);
            float[] levelMass = QuantileMath.probabilityMass(config.quantileLevels());
            float[] weights = new float[paths * q];
            for (int n = 0; n < paths; n++) {
                for (int k = 0; k < q; k++) {
                    weights[n * q + k] = pathMass[n] * levelMass[k];
                }
            }
            while (remaining > 0) {
                Unrolled u = unroll(context, group, future, prediction, unrolled, weights, outputPatches(remaining));
                prediction = u.prediction();
                context = u.context();
                future = u.future();
                predictions.add(prediction);
                remaining -= prediction[0][0].length;
            }
        }
        List<Forecast> out = new ArrayList<>(batch.size());
        float[] levels = config.quantileLevels();
        for (int s = 0; s < batch.size(); s++) {
            int targets = batch.get(s).targets();
            float[][][] v = new float[targets][q][horizon];
            for (int t = 0; t < targets; t++) {
                for (int k = 0; k < q; k++) {
                    int off = 0;
                    for (float[][][] p : predictions) {
                        float[] src = p[firstRow[s] + t][k];
                        int n = Math.min(src.length, horizon - off);
                        if (n <= 0) {
                            break;
                        }
                        System.arraycopy(src, 0, v[t][k], off, n);
                        off += n;
                    }
                }
            }
            out.add(new Forecast(levels, v));
        }
        return out;
    }

    /** One unrolling step's merged forecast, and the extended contexts and remaining futures of its paths. */
    private record Unrolled(float[][][] prediction, float[][] context, float[][] future) {}

    private int outputPatches(int remaining) {
        int ps = config.patchSize();
        return Math.min((remaining + ps - 1) / ps, options.maxOutputPatches());
    }

    /**
     * {@code _autoregressive_unroll_for_long_horizon}: extends each path's context with its level of
     * {@code prediction} ({@code [rows][Q][h]}, merged), slides the known covariates into it, runs the
     * next pass and merges its {@code paths · Q} forecasts per row back into the quantile levels.
     */
    private Unrolled unroll(float[][] context, int[] group, float[][] future, float[][][] prediction, float[] unrolled,
            float[] weights, int outputPatches) {
        int rows = prediction.length, paths = unrolled.length, q = config.numQuantiles(), h = prediction[0][0].length;
        float[] levels = config.quantileLevels();
        // interpolate_quantiles over (row, step): the unrolled levels of the merged forecast
        float[][] values = new float[rows * h][q];
        float[][] lv = new float[rows * h][];
        for (int b = 0; b < rows; b++) {
            for (int t = 0; t < h; t++) {
                for (int k = 0; k < q; k++) {
                    values[b * h + t][k] = prediction[b][k][t];
                }
                lv[b * h + t] = levels;
            }
        }
        float[][] up = QuantileMath.interpolate(unrolled, lv, values);
        int cap = config.contextLength();
        int len = Math.min(context[0].length + h, cap);
        float[][] next = new float[context.length][len];
        float[][] nextFuture = new float[future.length][];
        for (int b = 0; b < rows; b++) {
            for (int n = 0; n < paths; n++) {
                int r = b * paths + n;
                float[] row = new float[context[r].length + h];
                System.arraycopy(context[r], 0, row, 0, context[r].length);
                for (int t = 0; t < h; t++) {
                    row[context[r].length + t] = up[b * h + t][n];
                }
                System.arraycopy(row, row.length - len, next[r], 0, len);
                // _slide_context_and_future_covariates: known future values replace the forecast
                float[] f = future[r];
                int slide = Math.min(h, f.length);
                for (int t = 0; t < slide; t++) {
                    if (!Float.isNaN(f[t])) {
                        next[r][len - h + t] = f[t];
                    }
                }
                nextFuture[r] = Arrays.copyOfRange(f, Math.min(h, f.length), f.length);
            }
        }
        float[][][] pred = predictStep(next, group, nextFuture, outputPatches);
        int steps = pred[0][0].length;
        float[][] samples = new float[rows * steps][paths * q];
        for (int b = 0; b < rows; b++) {
            for (int n = 0; n < paths; n++) {
                for (int k = 0; k < q; k++) {
                    float[] src = pred[b * paths + n][k];
                    for (int t = 0; t < steps; t++) {
                        samples[b * steps + t][n * q + k] = src[t];
                    }
                }
            }
        }
        float[][] merged = QuantileMath.weightedQuantile(levels, weights, samples);
        float[][][] out = new float[rows][q][steps];
        for (int b = 0; b < rows; b++) {
            for (int t = 0; t < steps; t++) {
                for (int k = 0; k < q; k++) {
                    out[b][k][t] = merged[b * steps + t][k];
                }
            }
        }
        return new Unrolled(out, next, nextFuture);
    }

    /** {@code _predict_step}: the future covariates padded with {@code NaN} or cut to the forecast, then the model. */
    private float[][][] predictStep(float[][] context, int[] group, float[][] future, int outputPatches) {
        int size = outputPatches * config.patchSize();
        float[][] f = new float[future.length][size];
        for (int i = 0; i < f.length; i++) {
            int n = Math.min(size, future[i].length);
            System.arraycopy(future[i], 0, f[i], 0, n);
            Arrays.fill(f[i], n, size, Float.NaN);
        }
        return run(new ModelInput(context, group, f, outputPatches));
    }

    /**
     * The forward pass, as several when it holds more than {@code maxTokens} tokens: whole groups
     * per pass, each pass with the same context length, so the forecasts are unchanged.
     */
    private float[][][] run(ModelInput in) {
        int ps = config.patchSize();
        long perRow = (Math.min(in.length(), config.contextLength()) + ps - 1) / ps + (config.useRegToken() ? 1 : 0)
                + in.outputPatches();
        if (perRow * in.rows() <= maxTokens) {
            return forward.forward(in);
        }
        Map<Integer, List<Integer>> byGroup = new LinkedHashMap<>();
        for (int i = 0; i < in.rows(); i++) {
            byGroup.computeIfAbsent(in.group()[i], g -> new ArrayList<>()).add(i);
        }
        float[][][] out = new float[in.rows()][][];
        List<Integer> chunk = new ArrayList<>();
        for (List<Integer> members : byGroup.values()) {
            if (!chunk.isEmpty() && (chunk.size() + members.size()) * perRow > maxTokens) {
                runChunk(in, chunk, out);
                chunk.clear();
            }
            chunk.addAll(members);
        }
        runChunk(in, chunk, out);
        return out;
    }

    private void runChunk(ModelInput in, List<Integer> rows, float[][][] out) {
        int[] idx = rows.stream().mapToInt(Integer::intValue).sorted().toArray();
        float[][] c = new float[idx.length][], f = new float[idx.length][];
        int[] g = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            c[i] = in.context()[idx[i]];
            f[i] = in.futureCovariates()[idx[i]];
            g[i] = in.group()[idx[i]];
        }
        float[][][] r = forward.forward(new ModelInput(c, g, f, in.outputPatches()));
        for (int i = 0; i < idx.length; i++) {
            out[idx[i]] = r[i];
        }
    }
}
