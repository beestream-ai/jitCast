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
package ai.beestream.jitcast.toto;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;

/**
 * Toto 2.0's forecasting around a {@link TotoForward}: what Datadog's GluonTS predictor does with the default
 * {@code Toto2GluonTSModelConfig}, in one forward pass per batch.
 *
 * <ul>
 * <li>Each series' targets are forward-filled (leading gaps become 0) and its past covariates zero-filled;
 * both keep a mask of the observed steps. Past covariates are extra variates the targets attend to.</li>
 * <li>The context is trimmed to {@code maxContext} minus the horizon's patches but one, rounded down to
 * whole patches, and padded on the left to whole patches. The last context patch counts as observed
 * throughout.</li>
 * <li>A causal scaler normalizes every row: the running mean and (unbiased) standard deviation of the
 * observed steps, read at the end of each patch. Leading patches with fewer than {@code minObservations}
 * observed steps so far use the statistics of the first {@code minObservations} instead.</li>
 * <li>The scaled values go through {@code asinh}; the network's quantiles come back through {@code sinh}
 * and the scaler, then are capped to the observed range widened by {@code quantileCap} times the last
 * context scale, and sorted.</li>
 * </ul>
 */
public final class TotoPipeline {

    /** The series per forward pass by default. */
    public static final int DEFAULT_BATCH_SIZE = 64;

    private static final double MIN_SCALE = 1e-6;

    /**
     * Forecasting settings.
     *
     * @param maxContext the longest context plus horizon, in time steps (4096 in Datadog's evaluations)
     * @param minObservations the observations a leading patch needs for its own scaler statistics; 0 turns
     *        the fallback off
     * @param quantileCap how many last-context scales the quantiles may leave the observed range by; 0 turns
     *        the cap off
     */
    public record Options(int maxContext, int minObservations, double quantileCap) {

        /** Validates the settings. */
        public Options {
            if (maxContext < 1 || minObservations < 0 || !(quantileCap >= 0)) {
                throw new IllegalArgumentException("need maxContext ≥ 1, minObservations ≥ 0 and quantileCap ≥ 0");
            }
        }

        /** {@return the settings of Datadog's GluonTS predictor and evaluations} */
        public static Options defaults() {
            return new Options(4096, 8, 1e4);
        }

        /**
         * {@return these settings with another context limit}
         *
         * @param steps the longest context plus horizon
         */
        public Options withMaxContext(int steps) {
            return new Options(steps, minObservations, quantileCap);
        }
    }

    private final TotoConfig config;
    private final TotoForward forward;
    private final Options options;
    private final int batchSize;

    /**
     * A pipeline with the default settings.
     *
     * @param config the checkpoint's configuration
     * @param forward the network
     */
    public TotoPipeline(TotoConfig config, TotoForward forward) {
        this(config, forward, Options.defaults(), DEFAULT_BATCH_SIZE);
    }

    /**
     * A pipeline.
     *
     * @param config the checkpoint's configuration
     * @param forward the network
     * @param options the settings
     * @param batchSize the series per forward pass
     */
    public TotoPipeline(TotoConfig config, TotoForward forward, Options options, int batchSize) {
        if (batchSize < 1) {
            throw new IllegalArgumentException("batch size must be positive");
        }
        this.config = config;
        this.forward = forward;
        this.options = options;
        this.batchSize = batchSize;
    }

    /** {@return the settings} */
    public Options options() {
        return options;
    }

    /**
     * {@return this pipeline with other settings}
     *
     * @param o the settings
     */
    public TotoPipeline withOptions(Options o) {
        return new TotoPipeline(config, forward, o, batchSize);
    }

    /**
     * {@return this pipeline with another batch size}
     *
     * @param series the series per forward pass
     */
    public TotoPipeline withBatchSize(int series) {
        return new TotoPipeline(config, forward, options, series);
    }

    /**
     * Forecasts every series.
     *
     * @param series the inputs; targets and past covariates only
     * @param horizon the steps to forecast
     * @return one forecast per series, the nine levels 0.1 to 0.9
     */
    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        if (horizon < 1) {
            throw new IllegalArgumentException("horizon must be positive");
        }
        for (TimeSeries ts : series) {
            if (ts.futureCount() > 0 || ts.hasCategorical()) {
                throw new IllegalArgumentException("Toto 2.0 takes targets and numeric past covariates only");
            }
        }
        List<Forecast> out = new ArrayList<>(series.size());
        for (int from = 0; from < series.size(); from += batchSize) {
            out.addAll(forecastBatch(series.subList(from, Math.min(series.size(), from + batchSize)), horizon));
        }
        return out;
    }

    /**
     * The host side of one forward pass: the network's input and what turns its output into forecasts.
     *
     * @param input the network's input
     * @param target {@code [rows, length]}: the filled values, context then zeros
     * @param observed {@code [rows, length]}: the observed steps, the last context patch included
     * @param loc {@code [rows, length]}: the scaler's location per step, after the short-patch fallback
     * @param scale {@code [rows, length]}: its scale
     * @param context the context steps of every row
     * @param targets the target rows of each series (its first rows)
     */
    public record Prepared(TotoForward.Input input, float[] target, boolean[] observed, float[] loc, float[] scale, int context,
            int[] targets) {}

    /**
     * Builds the network's input for a batch.
     *
     * @param series the inputs
     * @param horizon the steps to forecast
     * @return the prepared batch
     */
    public Prepared prepare(List<TimeSeries> series, int horizon) {
        int p = config.patchSize();
        int forecastPatches = Math.ceilDiv(horizon, p);
        int ctxLimit = Math.max(p, (options.maxContext() - (forecastPatches - 1) * p) / p * p);
        int context = 0, rows = 0;
        int[] seriesStart = new int[series.size() + 1];
        int[] targets = new int[series.size()];
        for (int s = 0; s < series.size(); s++) {
            TimeSeries ts = series.get(s);
            context = Math.max(context, Math.ceilDiv(Math.min(ts.length(), ctxLimit), p) * p);
            rows += ts.targets() + ts.pastCount();
            seriesStart[s + 1] = rows;
            targets[s] = ts.targets();
        }
        int len = context + forecastPatches * p, n = len / p;
        float[] target = new float[rows * len];
        boolean[] observed = new boolean[rows * len];
        final int ctxCap = ctxLimit, ctxLen = context;
        IntStream.range(0, series.size()).parallel().forEach(si -> {
            TimeSeries ts = series.get(si);
            int r = seriesStart[si];
            int use = Math.min(ts.length(), ctxCap), skip = ts.length() - use, start = ctxLen - use;
            for (int kind = 0; kind < 2; kind++) {
                float[][] src = kind == 0 ? ts.target() : ts.pastCovariates();
                if (src == null) {
                    continue;
                }
                for (float[] row : src) {
                    float last = 0f;
                    if (kind == 0) {
                        for (int i = 0; i < skip; i++) {
                            last = Float.isNaN(row[i]) ? last : row[i];
                        }
                    }
                    for (int i = 0; i < use; i++) {
                        float v = row[skip + i];
                        boolean seen = !Float.isNaN(v);
                        if (seen) {
                            last = v;
                        }
                        target[r * len + start + i] = seen ? v : kind == 0 ? last : 0f;
                        observed[r * len + start + i] = seen;
                    }
                    Arrays.fill(observed, r * len + ctxLen - p, r * len + ctxLen, true);
                    r++;
                }
            }
        });
        float[] loc = new float[rows * len], scale = new float[rows * len];
        float[] values = new float[rows * len], missing = new float[rows * len];
        boolean[] valid = new boolean[rows * n];
        IntStream.range(0, rows).parallel().forEach(r -> {
            scaler(target, observed, r * len, len, p, loc, scale);
            if (options.minObservations() > 0) {
                backfill(target, observed, r * len, len, p, options.minObservations(), loc, scale);
            }
            for (int i = 0; i < len; i++) {
                int o = r * len + i;
                float z = observed[o] ? (target[o] - loc[o]) / scale[o] : 0f;
                values[o] = (float) asinh(z);
                missing[o] = observed[o] ? 0f : 1f;
            }
            for (int s = 0; s < n; s++) {
                boolean any = s * p >= ctxLen;
                for (int j = 0; j < p && !any; j++) {
                    any = observed[r * len + s * p + j];
                }
                valid[r * n + s] = any;
            }
        });
        TotoForward.Input input = new TotoForward.Input(rows, n, forecastPatches, values, missing, valid, seriesStart);
        return new Prepared(input, target, observed, loc, scale, context, targets);
    }

    private List<Forecast> forecastBatch(List<TimeSeries> series, int horizon) {
        Prepared prep = prepare(series, horizon);
        float[] head = forward.forward(prep.input());
        return finish(prep, head, horizon);
    }

    /**
     * Turns the network's output into forecasts.
     *
     * @param prep the prepared batch
     * @param head the network's output for it
     * @param horizon the steps to forecast
     * @return one forecast per series
     */
    public List<Forecast> finish(Prepared prep, float[] head, int horizon) {
        TotoForward.Input in = prep.input();
        int p = config.patchSize(), nq = config.quantiles(), k = in.forecast(), len = in.patches() * p, ctx = prep.context();
        float[][] real = new float[in.rows()][];
        IntStream.range(0, in.rows()).parallel().forEach(r -> {
            float[] q = new float[nq * k * p];
            for (int f = 0; f < k; f++) {
                for (int j = 0; j < p; j++) {
                    int o = r * len + ctx + f * p + j;
                    for (int l = 0; l < nq; l++) {
                        float z = head[((r * k + f) * p + j) * nq + l];
                        q[(l * k + f) * p + j] = (float) Math.sinh(z) * prep.scale()[o] + prep.loc()[o];
                    }
                }
            }
            clampNonFinite(q, nq * k, p);
            if (options.quantileCap() > 0) {
                double max = Double.NEGATIVE_INFINITY, min = Double.POSITIVE_INFINITY;
                for (int i = 0; i < ctx; i++) {
                    if (prep.observed()[r * len + i]) {
                        max = Math.max(max, prep.target()[r * len + i]);
                        min = Math.min(min, prep.target()[r * len + i]);
                    }
                }
                float anchor = (float) (options.quantileCap() * prep.scale()[r * len + ctx - 1]);
                float hi = (float) (max == Double.NEGATIVE_INFINITY ? 0 : max) + anchor;
                float lo = (float) (min == Double.POSITIVE_INFINITY ? 0 : min) - anchor;
                for (int i = 0; i < q.length; i++) {
                    q[i] = Math.min(Math.max(q[i], lo), hi);
                }
            }
            float[] col = new float[nq];
            for (int t = 0; t < k * p; t++) {
                for (int l = 0; l < nq; l++) {
                    col[l] = q[l * k * p + t];
                }
                Arrays.sort(col);
                for (int l = 0; l < nq; l++) {
                    q[l * k * p + t] = col[l];
                }
            }
            real[r] = q;
        });
        List<Forecast> out = new ArrayList<>(in.series());
        for (int s = 0; s < in.series(); s++) {
            int r0 = in.seriesStart()[s];
            float[][][] v = new float[prep.targets()[s]][nq][horizon];
            for (int t = 0; t < v.length; t++) {
                for (int l = 0; l < nq; l++) {
                    System.arraycopy(real[r0 + t], l * k * p, v[t][l], 0, horizon);
                }
            }
            out.add(new Forecast(TotoConfig.QUANTILES.clone(), v));
        }
        return out;
    }

    /** Infinities become the largest (smallest) finite value of their patch, as the reference does. */
    private static void clampNonFinite(float[] q, int patches, int p) {
        for (int g = 0; g < patches; g++) {
            float max = Float.NEGATIVE_INFINITY, min = Float.POSITIVE_INFINITY;
            for (int j = 0; j < p; j++) {
                float v = q[g * p + j];
                if (Float.isFinite(v)) {
                    max = Math.max(max, v);
                    min = Math.min(min, v);
                }
            }
            for (int j = 0; j < p; j++) {
                float v = q[g * p + j];
                if (v == Float.POSITIVE_INFINITY) {
                    q[g * p + j] = max;
                } else if (v == Float.NEGATIVE_INFINITY) {
                    q[g * p + j] = min;
                }
            }
        }
    }

    /**
     * The causal scaler of one row: running mean and unbiased variance of the observed steps (Welford's
     * increments, in double), each patch taking the statistics at its last step.
     */
    static void scaler(float[] x, boolean[] obs, int off, int len, int p, float[] loc, float[] scale) {
        double sum = 0, m2 = 0, mean = 0;
        int count = 0;
        for (int i = 0; i < len; i++) {
            double prev = mean;
            if (obs[off + i]) {
                sum += x[off + i];
                count++;
            }
            mean = sum / Math.max(count, 1);
            if (obs[off + i]) {
                m2 += (x[off + i] - prev) * (x[off + i] - mean);
            }
            if (i % p == p - 1) {
                double sd = Math.max(Math.sqrt(m2 / Math.max(count - 1, 1)), MIN_SCALE);
                Arrays.fill(loc, off + i - p + 1, off + i + 1, (float) mean);
                Arrays.fill(scale, off + i - p + 1, off + i + 1, (float) sd);
            }
        }
    }

    /**
     * Leading patches whose observations so far number fewer than {@code minObs} take the mean and unbiased
     * standard deviation of the row's first {@code minObs} observations.
     */
    static void backfill(float[] x, boolean[] obs, int off, int len, int p, int minObs, float[] loc, float[] scale) {
        double sum = 0;
        int n = 0;
        for (int i = 0; i < len && n < minObs; i++) {
            if (obs[off + i]) {
                sum += x[off + i];
                n++;
            }
        }
        float donorLoc = (float) (sum / Math.max(n, 1));
        double var = 0;
        int seen = 0;
        for (int i = 0; i < len && seen < n; i++) {
            if (obs[off + i]) {
                double dv = x[off + i] - donorLoc;
                var += dv * dv;
                seen++;
            }
        }
        float donorScale = (float) Math.max(Math.sqrt(var / Math.max(n - 1, 1)), MIN_SCALE);
        int cumulative = 0;
        for (int s = 0; s < len / p; s++) {
            for (int j = 0; j < p; j++) {
                if (obs[off + s * p + j]) {
                    cumulative++;
                }
            }
            if (cumulative >= minObs) {
                break;
            }
            Arrays.fill(loc, off + s * p, off + (s + 1) * p, donorLoc);
            Arrays.fill(scale, off + s * p, off + (s + 1) * p, donorScale);
        }
    }

    /** {@code asinh}, accurate near 0 too. */
    static double asinh(double x) {
        double a = Math.abs(x);
        double y = a > 1e8 ? Math.log(a) + Math.log(2) : Math.log1p(a + a * a / (1 + Math.sqrt(1 + a * a)));
        return Math.copySign(y, x);
    }
}
