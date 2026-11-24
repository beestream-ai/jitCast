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
package ai.beestream.jitcast.timesfm;

import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;

/**
 * TimesFM 2.5's {@code forecast()} around any {@link PatchForward}: per series, the leading NaNs
 * dropped and the gaps linearly interpolated, the context cut to {@code maxContext} and left-padded to
 * it; optionally the whole context normalized; then the decoding (each patch normalized by the running
 * statistics of the patches up to it, the forecast fed back 128 steps at a time), averaged with the
 * decoding of the negated series, the continuous quantile head, quantile crossing fixed, and
 * non-negative series clamped at 0, as the reference's {@code ForecastConfig} options choose.
 *
 * <p>TimesFM forecasts one series at a time: every target of a {@link TimeSeries} is forecast on its own,
 * and series with covariates are rejected. A series' forecast does not depend on the rest of its batch.
 * The reference's cached decoding is replaced by running every step over the whole sequence, which
 * gives the same outputs because attention is causal.
 */
public final class TimesFmPipeline {

    /** The most series one forward pass takes, by default. */
    public static final int DEFAULT_BATCH_SIZE = 64;

    /** The reference's floor below which a standard deviation counts as zero. */
    private static final float TOLERANCE = 1e-6f;

    /** {@return the most tokens one forward pass takes: {@code -Dtimesfm.maxTokens}, default 131,072} */
    public static long defaultMaxTokens() {
        return ai.beestream.jitcast.util.Options.number("timesfm.maxTokens", 131_072L, 1);
    }

    /**
     * The forecaster's settings, the reference's {@code ForecastConfig}. The defaults are the configuration
     * TimesFM's documentation recommends.
     *
     * @param maxContext context cap in time steps (the most recent are kept), rounded up to whole patches
     * @param normalizeInputs normalize each series by its context's mean and deviation first
     * @param continuousQuantileHead take the quantiles from the continuous quantile head, anchored to the
     *     median; horizons are then limited to the head's length (1,024 steps)
     * @param flipInvariance average with the negated forecast of the negated series
     * @param inferPositive clamp at 0 the forecasts of series whose context is non-negative
     * @param fixQuantileCrossing make the quantiles monotone outward from the median
     */
    public record Options(int maxContext, boolean normalizeInputs, boolean continuousQuantileHead, boolean flipInvariance,
            boolean inferPositive, boolean fixQuantileCrossing) {

        /**
         * {@return the recommended settings}
         *
         * @param c the checkpoint's configuration
         */
        public static Options defaults(TimesFmConfig c) {
            return new Options(1024, true, true, true, true, true);
        }

        /**
         * {@return these settings with another context cap}
         *
         * @param steps the most recent time steps to keep
         */
        public Options withMaxContext(int steps) {
            return new Options(steps, normalizeInputs, continuousQuantileHead, flipInvariance, inferPositive, fixQuantileCrossing);
        }

        /**
         * {@return these settings with input normalization on or off}
         *
         * @param on whether to normalize each series first
         */
        public Options withNormalizeInputs(boolean on) {
            return new Options(maxContext, on, continuousQuantileHead, flipInvariance, inferPositive, fixQuantileCrossing);
        }

        /**
         * {@return these settings with the continuous quantile head on or off}
         *
         * @param on whether to take the quantiles from the continuous head
         */
        public Options withContinuousQuantileHead(boolean on) {
            return new Options(maxContext, normalizeInputs, on, flipInvariance, inferPositive, fixQuantileCrossing);
        }

        /**
         * {@return these settings with flip averaging on or off}
         *
         * @param on whether to average with the forecast of the negated series
         */
        public Options withFlipInvariance(boolean on) {
            return new Options(maxContext, normalizeInputs, continuousQuantileHead, on, inferPositive, fixQuantileCrossing);
        }

        /**
         * {@return these settings with non-negative clamping on or off}
         *
         * @param on whether to clamp the forecasts of non-negative series at 0
         */
        public Options withInferPositive(boolean on) {
            return new Options(maxContext, normalizeInputs, continuousQuantileHead, flipInvariance, on, fixQuantileCrossing);
        }

        /**
         * {@return these settings with quantile-crossing fixing on or off}
         *
         * @param on whether to make the quantiles monotone
         */
        public Options withFixQuantileCrossing(boolean on) {
            return new Options(maxContext, normalizeInputs, continuousQuantileHead, flipInvariance, inferPositive, on);
        }
    }

    private final TimesFmConfig config;
    private final PatchForward forward;
    private final Options options;
    private final int batchSize;
    private final long maxTokens;

    /**
     * A pipeline with the recommended settings, the default batch size and the default token budget.
     *
     * @param config the checkpoint's configuration
     * @param forward the forward pass to run
     */
    public TimesFmPipeline(TimesFmConfig config, PatchForward forward) {
        this(config, forward, Options.defaults(config), DEFAULT_BATCH_SIZE, defaultMaxTokens());
    }

    /**
     * A pipeline with explicit settings.
     *
     * @param config the checkpoint's configuration
     * @param forward the forward pass to run
     * @param options the forecaster's settings
     * @param batchSize the most series per forward pass
     * @param maxTokens the most tokens (series × patches) per forward pass; a longer series runs alone
     * @throws IllegalArgumentException if a size is out of range
     */
    public TimesFmPipeline(TimesFmConfig config, PatchForward forward, Options options, int batchSize, long maxTokens) {
        if (batchSize < 1 || maxTokens < 1) {
            throw new IllegalArgumentException("batchSize and maxTokens must be >= 1");
        }
        if (options.maxContext() < 1 || options.maxContext() > config.contextLength()) {
            throw new IllegalArgumentException("maxContext must be in [1, " + config.contextLength() + "], got " + options.maxContext());
        }
        this.config = config;
        this.forward = forward;
        this.options = options;
        this.batchSize = batchSize;
        this.maxTokens = maxTokens;
    }

    /** {@return the forecaster's settings} */
    public Options options() {
        return options;
    }

    /**
     * {@return the same pipeline with other settings}
     *
     * @param o the settings
     */
    public TimesFmPipeline withOptions(Options o) {
        return new TimesFmPipeline(config, forward, o, batchSize, maxTokens);
    }

    /**
     * {@return the same pipeline with another batch size}
     *
     * @param series the most series per forward pass
     */
    public TimesFmPipeline withBatchSize(int series) {
        return new TimesFmPipeline(config, forward, options, series, maxTokens);
    }

    /**
     * {@return the same pipeline with another token budget per forward pass}
     *
     * @param tokens the most tokens per forward pass
     */
    public TimesFmPipeline withMaxTokens(long tokens) {
        return new TimesFmPipeline(config, forward, options, batchSize, tokens);
    }

    /** The padded context length: {@code maxContext} rounded up to whole patches. */
    int context() {
        int p = config.patchLength();
        return (options.maxContext() + p - 1) / p * p;
    }

    /** The decoded length for a horizon: whole output patches. */
    int decodedLength(int horizon) {
        int o = config.horizonLength();
        return (horizon + o - 1) / o * o;
    }

    /**
     * Forecasts every series, as {@link ai.beestream.jitcast.api.ForecastEngine#forecast}.
     *
     * @param series the inputs
     * @param horizon the number of steps to forecast, at least 1
     * @return one forecast per series, in order
     * @throws IllegalArgumentException if {@code horizon} is out of range or a series has covariates
     */
    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        if (horizon < 1) {
            throw new IllegalArgumentException("horizon must be >= 1, got " + horizon);
        }
        int len = decodedLength(horizon);
        if (context() + len > config.contextLength()) {
            throw new IllegalArgumentException("maxContext + horizon (" + context() + " + " + len + ") exceeds the model's "
                    + config.contextLength() + " steps");
        }
        if (options.continuousQuantileHead() && len > config.quantileHorizonLength()) {
            throw new IllegalArgumentException("the continuous quantile head covers " + config.quantileHorizonLength()
                    + " steps; for a horizon of " + horizon + " turn it off (Options.withContinuousQuantileHead(false))");
        }
        List<float[]> rows = new ArrayList<>();
        for (TimeSeries ts : series) {
            if (ts.pastCount() + ts.futureCount() > 0 || ts.hasCategorical()) {
                throw new IllegalArgumentException("TimesFM does not take covariates");
            }
            for (float[] target : ts.target()) {
                rows.add(target);
            }
        }
        float[][][] out = new float[rows.size()][][];
        int ctxPatches = context() / config.patchLength();
        int tokensPerRow = ctxPatches + (len / config.horizonLength() - 1) * config.patchesPerOutput();
        int perPass = (int) Math.max(1, Math.min(batchSize, maxTokens / tokensPerRow));
        for (int start = 0; start < rows.size(); start += perPass) {
            List<float[]> window = rows.subList(start, Math.min(rows.size(), start + perPass));
            float[][][] fc = forecastWindow(window, horizon);
            System.arraycopy(fc, 0, out, start, fc.length);
        }
        float[] levels = config.quantileLevels();
        List<Forecast> forecasts = new ArrayList<>(series.size());
        int r = 0;
        for (TimeSeries ts : series) {
            float[][][] values = new float[ts.targets()][][];
            for (int t = 0; t < values.length; t++) {
                values[t] = out[r++];
            }
            forecasts.add(new Forecast(levels.clone(), values));
        }
        return forecasts;
    }

    /**
     * The batch the reference assembles for decoding.
     *
     * @param inputs {@code [rows][context]} the values, after normalization when that is on
     * @param masks {@code [rows][context]} 1 where a step is padding
     * @param positive whether each row's context is non-negative
     * @param mu each row's normalization mean (0 without normalization)
     * @param sigma each row's normalization deviation (1 without normalization)
     */
    record Assembled(float[][] inputs, float[][] masks, boolean[] positive, float[] mu, float[] sigma) {}

    /** Drops leading NaNs, interpolates the rest, cuts or pads to the context, and normalizes. */
    Assembled assemble(List<float[]> rows) {
        int c = context(), n = rows.size();
        float[][] inputs = new float[n][c], masks = new float[n][c];
        boolean[] positive = new boolean[n];
        float[] mu = new float[n], sigma = new float[n];
        for (int r = 0; r < n; r++) {
            float[] v = interpolate(stripLeadingNaNs(rows.get(r)));
            int w = v.length;
            if (w >= c) {
                System.arraycopy(v, w - c, inputs[r], 0, c);
            } else {
                System.arraycopy(v, 0, inputs[r], c - w, w);
                for (int i = 0; i < c - w; i++) {
                    masks[r][i] = 1f;
                }
            }
            boolean pos = true;
            for (float x : inputs[r]) {
                pos &= x >= 0;
            }
            positive[r] = pos;
            sigma[r] = 1f;
            if (options.normalizeInputs()) {
                double mag = 1;
                for (float x : inputs[r]) {
                    mag = Math.max(mag, Math.abs(x));
                }
                float magnitude = (float) mag;
                double sum = 0;
                for (float x : inputs[r]) {
                    sum += x / magnitude;
                }
                double mean = sum / c, ss = 0;
                for (float x : inputs[r]) {
                    double dv = x / magnitude - mean;
                    ss += dv * dv;
                }
                mu[r] = (float) (mean * magnitude);
                sigma[r] = (float) (Math.sqrt(ss / (c - 1)) * magnitude);
                float safe = safe(sigma[r]);
                for (int i = 0; i < c; i++) {
                    inputs[r][i] = (inputs[r][i] - mu[r]) / safe;
                }
            }
        }
        return new Assembled(inputs, masks, positive, mu, sigma);
    }

    private static float safe(float sigma) {
        return sigma < TOLERANCE ? 1f : sigma;
    }

    private static float[] stripLeadingNaNs(float[] v) {
        int s = 0;
        while (s < v.length && Float.isNaN(v[s])) {
            s++;
        }
        float[] out = new float[v.length - s];
        System.arraycopy(v, s, out, 0, out.length);
        return out;
    }

    /** NaNs filled by linear interpolation, as {@code numpy.interp}: past the last value it is repeated. */
    private static float[] interpolate(float[] v) {
        float[] out = v.clone();
        int last = -1;
        for (int i = 0; i < out.length; i++) {
            if (Float.isNaN(out[i])) {
                continue;
            }
            if (last >= 0 && i - last > 1) {
                for (int j = last + 1; j < i; j++) {
                    out[j] = (float) (out[last] + (out[i] - (double) out[last]) * (j - last) / (i - last));
                }
            }
            last = i;
        }
        if (last >= 0) {
            for (int j = last + 1; j < out.length; j++) {
                out[j] = out[last];
            }
        }
        return out;
    }

    /**
     * One decoding of the reference ({@code TimesFM_2p5_200M_torch_module.decode}), renormalized.
     *
     * @param last {@code [rows][horizonLength][outputs]} the prefill's last patch
     * @param spread {@code [rows][quantileHorizonLength][outputs]} its continuous quantile head
     * @param ar {@code [rows][steps][horizonLength][outputs]} each fed-back step's last patch
     */
    record Decoded(float[][][] last, float[][][] spread, float[][][][] ar) {}

    /** Running statistics: the count, mean and deviation of the observed steps so far. */
    private static final class Stats {
        double n, mu, sigma;

        /** Adds one patch, as the reference's {@code update_running_stats}. */
        void add(float[] x, int off, float[] mask, int p) {
            double incN = 0, sum = 0;
            for (int i = 0; i < p; i++) {
                if (mask == null || mask[off + i] < 0.5f) {
                    incN++;
                    sum += x[off + i];
                }
            }
            double incMu = incN == 0 ? 0 : sum / incN, ss = 0;
            for (int i = 0; i < p; i++) {
                if (mask == null || mask[off + i] < 0.5f) {
                    double dv = x[off + i] - incMu;
                    ss += dv * dv;
                }
            }
            double incSigma = Math.sqrt(incN == 0 ? 0 : ss / incN);
            double newN = n + incN;
            double newMu = newN == 0 ? 0 : (n * mu + incMu * incN) / newN;
            double var = newN == 0 ? 0
                    : (n * sigma * sigma + incN * incSigma * incSigma + n * (mu - newMu) * (mu - newMu)
                            + incN * (incMu - newMu) * (incMu - newMu)) / newN;
            n = newN;
            mu = newMu;
            sigma = Math.sqrt(Math.max(var, 0));
        }
    }

    /** Decodes {@code steps} fed-back output patches after the prefill. */
    Decoded decode(float[][] inputs, float[][] masks, int steps) {
        int rows = inputs.length, p = config.patchLength(), c = inputs[0].length, n = c / p;
        int o = config.horizonLength(), q = config.outputs(), qs = config.quantileHorizonLength(), m = config.patchesPerOutput();
        // patches that are padding in every row change no output: skip them
        int lead = n;
        for (float[] mask : masks) {
            int full = 0;
            while (full < n && mask[(full + 1) * p - 1] > 0.5f) {
                full++;
            }
            lead = Math.min(lead, full);
        }
        lead = Math.min(lead, n - 1);
        int patches = n - lead, total = patches + steps * m;
        float[] values = new float[rows * total * p], mask = new float[rows * total * p];
        Stats[] stats = new Stats[rows];
        float[] lastMu = new float[rows], lastSigma = new float[rows];
        for (int r = 0; r < rows; r++) {
            stats[r] = new Stats();
            for (int t = 0; t < patches; t++) {
                int src = (lead + t) * p, dst = (r * total + t) * p;
                stats[r].add(inputs[r], src, masks[r], p);
                float mu = (float) stats[r].mu, s = safe((float) stats[r].sigma);
                for (int i = 0; i < p; i++) {
                    boolean pad = masks[r][src + i] > 0.5f;
                    mask[dst + i] = pad ? 1f : 0f;
                    values[dst + i] = pad ? 0f : (inputs[r][src + i] - mu) / s;
                }
            }
            lastMu[r] = (float) stats[r].mu;
            lastSigma[r] = (float) stats[r].sigma;
        }
        // the statistics each run's last patch is renormalized with: the prefill's, then each step's
        float[][] runMu = new float[steps + 1][], runSigma = new float[steps + 1][];
        runMu[0] = lastMu.clone();
        runSigma[0] = lastSigma.clone();
        PatchForward.Feeder feeder = (prevHeads, step) -> {
            float[] fedValues = new float[rows * m * p], fedMasks = new float[rows * m * p];
            for (int r = 0; r < rows; r++) {
                float s = safe(runSigma[step][r]);
                for (int j = 0; j < m; j++) {
                    int src = j * p;
                    float[] fed = new float[p];
                    for (int i = 0; i < p; i++) {
                        fed[i] = prevHeads.point()[(r * o + src + i) * q + TimesFmConfig.MEDIAN_OUTPUT] * s + runMu[step][r];
                    }
                    stats[r].add(fed, 0, null, p);
                    float mu = (float) stats[r].mu, sd = safe((float) stats[r].sigma);
                    for (int i = 0; i < p; i++) {
                        fedValues[(r * m + j) * p + i] = (fed[i] - mu) / sd;
                    }
                }
            }
            runMu[step + 1] = new float[rows];
            runSigma[step + 1] = new float[rows];
            for (int r = 0; r < rows; r++) {
                runMu[step + 1][r] = (float) stats[r].mu;
                runSigma[step + 1][r] = (float) stats[r].sigma;
            }
            return new PatchForward.Batch(rows, m, fedValues, fedMasks, false);
        };
        List<PatchForward.Heads> runs = forward.decode(new PatchForward.Batch(rows, patches, compact(values, rows, total, patches, p),
                compact(mask, rows, total, patches, p), true), steps, feeder);
        PatchForward.Heads h = runs.get(0);
        float[][][] last = new float[rows][o][q], spread = new float[rows][qs][q];
        for (int r = 0; r < rows; r++) {
            float s = safe(runSigma[0][r]);
            for (int t = 0; t < o; t++) {
                for (int k = 0; k < q; k++) {
                    last[r][t][k] = h.point()[(r * o + t) * q + k] * s + runMu[0][r];
                }
            }
            for (int t = 0; t < qs; t++) {
                for (int k = 0; k < q; k++) {
                    spread[r][t][k] = h.spread()[(r * qs + t) * q + k] * s + runMu[0][r];
                }
            }
        }
        float[][][][] ar = new float[rows][steps][o][q];
        for (int step = 0; step < steps; step++) {
            PatchForward.Heads hs = runs.get(step + 1);
            for (int r = 0; r < rows; r++) {
                float s = safe(runSigma[step + 1][r]);
                for (int t = 0; t < o; t++) {
                    for (int k = 0; k < q; k++) {
                        ar[r][step][t][k] = hs.point()[(r * o + t) * q + k] * s + runMu[step + 1][r];
                    }
                }
            }
        }
        return new Decoded(last, spread, ar);
    }

    /** The first {@code patches} of each row's {@code total}-patch slot. */
    private static float[] compact(float[] all, int rows, int total, int patches, int p) {
        if (patches == total) {
            return all;
        }
        float[] out = new float[rows * patches * p];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(all, r * total * p, out, r * patches * p, patches * p);
        }
        return out;
    }

    /** {@code [rows][quantiles][horizon]} for one window of rows. */
    private float[][][] forecastWindow(List<float[]> rows, int horizon) {
        Assembled a = assemble(rows);
        int n = rows.size(), len = decodedLength(horizon), o = config.horizonLength(), q = config.outputs();
        int steps = len / o - 1;
        Decoded d = decode(a.inputs(), a.masks(), steps);
        float[][][] full = stitch(d, n, len, o, q);
        float[][][] spread = d.spread();
        if (options.flipInvariance()) {
            float[][] neg = new float[n][];
            for (int r = 0; r < n; r++) {
                neg[r] = a.inputs()[r].clone();
                for (int i = 0; i < neg[r].length; i++) {
                    neg[r][i] = -neg[r][i];
                }
            }
            Decoded f = decode(neg, a.masks(), steps);
            float[][][] flipped = stitch(f, n, len, o, q);
            float[][][] flippedSpread = f.spread();
            for (int r = 0; r < n; r++) {
                average(full[r], flipped[r], q);
                average(spread[r], flippedSpread[r], q);
            }
        }
        int median = TimesFmConfig.MEDIAN_OUTPUT;
        if (options.continuousQuantileHead()) {
            for (int r = 0; r < n; r++) {
                for (int t = 0; t < len; t++) {
                    for (int k = 1; k < q; k++) {
                        if (k != median) {
                            full[r][t][k] = spread[r][t][k] - spread[r][t][median] + full[r][t][median];
                        }
                    }
                }
            }
        }
        float[][][] out = new float[n][q - 1][horizon];
        for (int r = 0; r < n; r++) {
            float s = safe(a.sigma()[r]);
            for (int t = 0; t < horizon; t++) {
                float[] v = full[r][t];
                if (options.fixQuantileCrossing()) {
                    for (int k = median - 1; k >= 1; k--) {
                        v[k] = Math.min(v[k], v[k + 1]);
                    }
                    for (int k = median + 1; k < q; k++) {
                        v[k] = Math.max(v[k], v[k - 1]);
                    }
                }
                for (int k = 1; k < q; k++) {
                    float x = options.normalizeInputs() ? v[k] * s + a.mu()[r] : v[k];
                    if (options.inferPositive() && a.positive()[r]) {
                        x = Math.max(x, 0f);
                    }
                    out[r][k - 1][t] = x;
                }
            }
        }
        return out;
    }

    /** The prefill's last patch followed by every fed-back step, {@code [rows][len][outputs]}. */
    private static float[][][] stitch(Decoded d, int rows, int len, int o, int q) {
        float[][][] full = new float[rows][len][];
        for (int r = 0; r < rows; r++) {
            for (int t = 0; t < len; t++) {
                float[] src = t < o ? d.last()[r][t] : d.ar()[r][t / o - 1][t % o];
                full[r][t] = src.clone();
            }
        }
        return full;
    }

    /** {@code x ← (x − flip(y)) / 2}, where {@code flip} keeps the mean output and reverses the quantiles. */
    private static void average(float[][] x, float[][] y, int q) {
        for (int t = 0; t < x.length; t++) {
            x[t][0] = (x[t][0] - y[t][0]) / 2;
            float[] xt = x[t].clone();
            for (int k = 1; k < q; k++) {
                x[t][k] = (xt[k] - y[t][q - k]) / 2;
            }
        }
    }
}
