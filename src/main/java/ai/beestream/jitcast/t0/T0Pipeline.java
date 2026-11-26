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
package ai.beestream.jitcast.t0;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;

/**
 * t0's {@code predict()} around any {@link T0Forward}. Every series becomes rows of the model's native input:
 * its targets, its past-only covariates (historical rows, withheld over the horizon like the targets) and its
 * known-future covariates, right-aligned with the other series of its batch. Optionally the missing stretch
 * before a series' first observation is turned to padding and known-future covariates constant over the
 * context are dropped. Each forward window is scaled by the causal scaler, and its patches before the
 * forecast region predict the quantiles of the patch after them. Horizons up to 1,024 steps take one pass;
 * longer ones continue autoregressively, one trajectory per quantile level, reduced back to the levels by
 * their weighted empirical distribution. Levels between the trained ones are interpolated, levels beyond them
 * extrapolated on exponential tails. Then, as the options choose, the forecast is averaged with that of the
 * negated series, clipped at zero and rounded to integers where the series' history is.
 *
 * <p>A series' forecast does not depend on the rest of its batch: series attend only to their own rows.
 */
public final class T0Pipeline {

    /** The most series one forward pass takes, by default. */
    public static final int DEFAULT_BATCH_SIZE = 64;

    /** {@return the most tokens (rows × patches) one forward pass takes: {@code -Dt0.maxTokens}, default 16,384} */
    public static long defaultMaxTokens() {
        return ai.beestream.jitcast.util.Options.number("t0.maxTokens", 16_384L, 1);
    }

    /**
     * The forecaster's settings, the reference's {@code predict()} keywords.
     *
     * @param quantileLevels the levels to forecast, ascending in (0, 1): levels between the trained ones are
     *     interpolated, levels beyond them extrapolated
     * @param trimLeadingMissing treat the missing stretch before a series' first observation as padding
     * @param symmetricAveraging average each level {@code q} with minus level {@code 1 − q} of the forecast of
     *     the negated series (two forward passes), except for series whose history has a minimum of exactly 0
     * @param clipNonnegative clip at 0 the forecasts of targets whose history is non-negative
     * @param snapIntegers round to integers the forecasts of targets whose history is integer-valued
     * @param dropConstantCovariates ignore a known-future covariate constant over its series' context
     */
    public record Options(double[] quantileLevels, boolean trimLeadingMissing, boolean symmetricAveraging,
            boolean clipNonnegative, boolean snapIntegers, boolean dropConstantCovariates) {

        /**
         * Checks the levels.
         *
         * @throws IllegalArgumentException unless the levels are ascending, unique and in (0, 1)
         */
        public Options {
            quantileLevels = T0Quantiles.validate(quantileLevels);
        }

        /**
         * {@return the reference's defaults, with the checkpoint's trained levels}: no adjustment, constant
         * known-future covariates dropped
         *
         * @param c the checkpoint's configuration
         */
        public static Options defaults(T0Config c) {
            return new Options(c.quantileLevels(), false, false, false, false, true);
        }

        /**
         * {@return these settings with all four adjustments the reference recommends turned on}: leading
         * missing values trimmed, symmetric averaging, non-negative clipping and integer snapping
         */
        public Options withRecommendedAdjustments() {
            return new Options(quantileLevels, true, true, true, true, dropConstantCovariates);
        }

        /**
         * {@return these settings with other quantile levels}
         *
         * @param levels the levels to forecast
         */
        public Options withQuantileLevels(double... levels) {
            return new Options(levels, trimLeadingMissing, symmetricAveraging, clipNonnegative, snapIntegers, dropConstantCovariates);
        }

        /**
         * {@return these settings with leading-missing trimming on or off}
         *
         * @param on whether to trim
         */
        public Options withTrimLeadingMissing(boolean on) {
            return new Options(quantileLevels, on, symmetricAveraging, clipNonnegative, snapIntegers, dropConstantCovariates);
        }

        /**
         * {@return these settings with symmetric averaging on or off}
         *
         * @param on whether to average with the negated series' forecast
         */
        public Options withSymmetricAveraging(boolean on) {
            return new Options(quantileLevels, trimLeadingMissing, on, clipNonnegative, snapIntegers, dropConstantCovariates);
        }

        /**
         * {@return these settings with non-negative clipping on or off}
         *
         * @param on whether to clip
         */
        public Options withClipNonnegative(boolean on) {
            return new Options(quantileLevels, trimLeadingMissing, symmetricAveraging, on, snapIntegers, dropConstantCovariates);
        }

        /**
         * {@return these settings with integer snapping on or off}
         *
         * @param on whether to round
         */
        public Options withSnapIntegers(boolean on) {
            return new Options(quantileLevels, trimLeadingMissing, symmetricAveraging, clipNonnegative, on, dropConstantCovariates);
        }

        /**
         * {@return these settings with constant known-future covariates dropped or kept}
         *
         * @param on whether to drop them
         */
        public Options withDropConstantCovariates(boolean on) {
            return new Options(quantileLevels, trimLeadingMissing, symmetricAveraging, clipNonnegative, snapIntegers, on);
        }
    }

    /** Sees every forward window, for the golden tests. */
    @FunctionalInterface
    interface Observer {
        /**
         * @param window the window the forward pass ran on
         * @param horizon the steps it predicted
         * @param out {@code [rows · horizon · trained levels]} its predictions in data space
         */
        void call(T0Batch window, int horizon, float[] out);
    }

    private final T0Config config;
    private final T0Forward forward;
    private final Options options;
    private final int batchSize;
    private final long maxTokens;
    private final T0Scaler scaler;
    private Observer observer;

    /**
     * A pipeline with the reference's default settings, the default batch size and the default token budget.
     *
     * @param config the checkpoint's configuration
     * @param forward the forward pass to run
     */
    public T0Pipeline(T0Config config, T0Forward forward) {
        this(config, forward, Options.defaults(config), DEFAULT_BATCH_SIZE, defaultMaxTokens());
    }

    /**
     * A pipeline with explicit settings.
     *
     * @param config the checkpoint's configuration
     * @param forward the forward pass to run
     * @param options the forecaster's settings
     * @param batchSize the most series per forward pass
     * @param maxTokens the most tokens (rows × patches) per forward pass; a larger series runs alone
     * @throws IllegalArgumentException if a size is out of range
     */
    public T0Pipeline(T0Config config, T0Forward forward, Options options, int batchSize, long maxTokens) {
        if (batchSize < 1 || maxTokens < 1) {
            throw new IllegalArgumentException("batchSize and maxTokens must be >= 1");
        }
        this.config = config;
        this.forward = forward;
        this.options = options;
        this.batchSize = batchSize;
        this.maxTokens = maxTokens;
        this.scaler = new T0Scaler(config);
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
    public T0Pipeline withOptions(Options o) {
        return new T0Pipeline(config, forward, o, batchSize, maxTokens);
    }

    /**
     * {@return the same pipeline with another batch size}
     *
     * @param series the most series per forward pass
     */
    public T0Pipeline withBatchSize(int series) {
        return new T0Pipeline(config, forward, options, series, maxTokens);
    }

    /**
     * {@return the same pipeline with another token budget per forward pass}
     *
     * @param tokens the most tokens per forward pass
     */
    public T0Pipeline withMaxTokens(long tokens) {
        return new T0Pipeline(config, forward, options, batchSize, tokens);
    }

    T0Pipeline observe(Observer o) {
        this.observer = o;
        return this;
    }

    /**
     * Forecasts every series, as {@link ai.beestream.jitcast.api.ForecastEngine#forecast}.
     *
     * @param series the inputs
     * @param horizon the number of steps to forecast, at least 1
     * @return one forecast per series, in order, at {@link Options#quantileLevels()}
     * @throws IllegalArgumentException if {@code horizon} is out of range or a series has categorical
     *     covariates or a known-future covariate shorter than its context plus the horizon
     */
    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        if (horizon < 1) {
            throw new IllegalArgumentException("horizon must be >= 1, got " + horizon);
        }
        int p = config.patchSize();
        int forecastPatches = Math.min(roundUp(horizon, p), T0Config.MAX_HORIZON) / p;
        int paths = horizon > T0Config.MAX_HORIZON
                ? T0Quantiles.rolloutLevels(config.quantileLevels(), levelsFor(options.quantileLevels())).length : 1;
        List<Forecast> out = new ArrayList<>(series.size());
        int start = 0;
        while (start < series.size()) {
            int end = start, rows = 0, longest = 0;
            while (end < series.size() && end - start < batchSize) {
                TimeSeries ts = series.get(end);
                int r = rows + ts.rows(), l = Math.max(longest, ts.length());
                long tokens = (long) r * paths * (roundUp(l, p) / p + forecastPatches);
                if (end > start && tokens > maxTokens) {
                    break;
                }
                rows = r;
                longest = l;
                end++;
            }
            out.addAll(forecastBatch(series.subList(start, end), horizon));
            start = end;
        }
        return out;
    }

    private double[] levelsFor(double[] requested) {
        return options.symmetricAveraging() ? withMirrors(requested) : requested;
    }

    /** One batch, as one {@code predict()} call on the batched native input. */
    private List<Forecast> forecastBatch(List<TimeSeries> series, int horizon) {
        T0Batch b = T0Batch.of(series, horizon);
        int context = b.contextEnd();
        if (options.trimLeadingMissing()) {
            b = b.trimLeadingMissing(context);
        }
        if (options.dropConstantCovariates() && b.hasFuture()) {
            b = b.dropConstantCovariates(context);
        }
        int[] targets = b.targetRows();
        double[] requested = options.quantileLevels();
        double[] levels = levelsFor(requested);
        float[] predictions = forecastOnce(b, horizon, context, levels, targets);
        int nq = requested.length;
        if (options.symmetricAveraging() || options.clipNonnegative() || options.snapIntegers()) {
            double[][] history = new double[targets.length][];
            for (int t = 0; t < targets.length; t++) {
                history[t] = b.history(targets[t], context);
            }
            double[] values = toDouble(predictions);
            if (options.symmetricAveraging()) {
                boolean any = false;
                for (double[] h : history) {
                    any |= averaged(h);
                }
                double[] negated = any ? toDouble(forecastOnce(b.negated(), horizon, context, levels, targets)) : values;
                values = symmetricAverage(values, negated, levels, requested, history, horizon);
            }
            bound(values, history, horizon, nq);
            predictions = new float[values.length];
            for (int i = 0; i < values.length; i++) {
                predictions[i] = (float) values[i];
            }
        }
        float[] out = predictions;
        float[] lv = new float[nq];
        for (int i = 0; i < nq; i++) {
            lv[i] = (float) requested[i];
        }
        List<Forecast> forecasts = new ArrayList<>(series.size());
        int t = 0;
        for (TimeSeries ts : series) {
            float[][][] values = new float[ts.targets()][nq][horizon];
            for (int k = 0; k < ts.targets(); k++, t++) {
                for (int h = 0; h < horizon; h++) {
                    for (int q = 0; q < nq; q++) {
                        values[k][q][h] = out[(t * horizon + h) * nq + q];
                    }
                }
            }
            forecasts.add(new Forecast(lv.clone(), values));
        }
        return forecasts;
    }

    /** {@code _forecast}: one rollout at {@code levels}, sanitized and extrapolated, {@code [targets · H · levels]}. */
    private float[] forecastOnce(T0Batch b, int horizon, int context, double[] levels, int[] targets) {
        double[] trained = config.quantileLevels();
        double[] rollout = T0Quantiles.rolloutLevels(trained, levels);
        float[] predictions = rollout(b, horizon, context, trained, rollout, targets);
        for (int i = 0; i < predictions.length; i++) {
            if (!Float.isFinite(predictions[i])) {
                predictions[i] = 0f;
            }
        }
        return T0Quantiles.extrapolate(levels, rollout, predictions, trained);
    }

    /** The reference's {@code rollout}: {@code [targets · H · query]}. */
    private float[] rollout(T0Batch series, int length, int context, double[] trained, double[] query, int[] targets) {
        int p = config.patchSize(), nt = trained.length, nq = query.length, tn = targets.length;
        T0Batch buffer = prepareBuffer(series, length, context);
        int contextWidth = roundUp(context, p);
        int horizon = Math.min(roundUp(length, p), T0Config.MAX_HORIZON);
        float[] block = selectRows(step(buffer.slice(0, contextWidth + horizon), horizon), targets, horizon * nt);
        float[] prediction = T0Quantiles.interpolate(query, trained, block);
        if (length <= horizon) {
            return cut(prediction, tn, horizon, length, nq);
        }
        int[] pathTargets = new int[tn * nq];
        for (int t = 0; t < tn; t++) {
            for (int j = 0; j < nq; j++) {
                pathTargets[t * nq + j] = targets[t] * nq + j;
            }
        }
        T0Batch paths = expand(buffer, nq);
        List<float[]> predictions = new ArrayList<>(List.of(prediction));
        List<Integer> widths = new ArrayList<>(List.of(horizon));
        int decoded = horizon, remaining = length - horizon;
        while (remaining > 0) {
            int previous = widths.get(widths.size() - 1);
            paths = update(paths, predictions.get(predictions.size() - 1), tn, previous, nq, contextWidth + decoded - previous);
            horizon = Math.min(roundUp(remaining, p), T0Config.MAX_HORIZON);
            float[] out = selectRows(step(paths.slice(decoded, contextWidth + decoded + horizon), horizon), pathTargets,
                    horizon * nt);
            // (t q) h pq -> t q pq h
            float[] traj = new float[out.length];
            for (int tq = 0; tq < tn * nq; tq++) {
                for (int h = 0; h < horizon; h++) {
                    for (int i = 0; i < nt; i++) {
                        traj[(tq * nt + i) * horizon + h] = out[(tq * horizon + h) * nt + i];
                    }
                }
            }
            predictions.add(T0Quantiles.reduceRollout(traj, tn, horizon, trained, query));
            widths.add(horizon);
            decoded += horizon;
            remaining -= horizon;
        }
        float[] all = new float[tn * length * nq];
        for (int t = 0; t < tn; t++) {
            int at = 0;
            for (int k = 0; k < predictions.size() && at < length; k++) {
                int w = widths.get(k), take = Math.min(w, length - at);
                System.arraycopy(predictions.get(k), t * w * nq, all, (t * length + at) * nq, take * nq);
                at += take;
            }
        }
        return all;
    }

    /** {@code prepare_rollout_buffer}: the context padded on the left to whole patches, then the forecast region. */
    private T0Batch prepareBuffer(T0Batch s, int length, int context) {
        int p = config.patchSize();
        int padLeft = Math.floorMod(-context, p), forecastWidth = roundUp(length, p);
        int known = Math.min(s.width() - context, forecastWidth);
        int w = padLeft + context + forecastWidth;
        T0Batch out = T0Batch.empty(s.rows(), w);
        IntStream all = IntStream.range(0, s.rows());
        (s.rows() > 1 && (long) s.rows() * w >= T0Scaler.PARALLEL_CELLS ? all.parallel() : all).forEach(r -> {
            int type = s.rowType(r), group = s.rowGroup(r);
            boolean future = type == T0Batch.FUTURE, withheld = type == T0Batch.TARGET || type == T0Batch.HISTORICAL;
            int src = r * s.width(), dst = r * w + padLeft;
            System.arraycopy(s.values(), src, out.values(), dst, context);
            System.arraycopy(s.mask(), src, out.mask(), dst, context);
            System.arraycopy(s.group(), src, out.group(), dst, context);
            System.arraycopy(s.type(), src, out.type(), dst, context);
            for (int i = 0; i < forecastWidth; i++) {
                int o = dst + context + i;
                out.values()[o] = 0f;
                out.mask()[o] = withheld ? T0Batch.WITHHELD : T0Batch.PAD;
                if (future && i < known) {
                    out.values()[o] = s.values()[src + context + i];
                    out.mask()[o] = s.mask()[src + context + i];
                }
                out.group()[o] = group;
                out.type()[o] = type;
            }
        });
        return out;
    }

    /** {@code expand_prediction_paths}: each row repeated per trajectory, with a group of its own. */
    private static T0Batch expand(T0Batch b, int n) {
        int w = b.width();
        T0Batch out = T0Batch.empty(b.rows() * n, w);
        for (int r = 0; r < b.rows(); r++) {
            int group = b.rowGroup(r), type = b.rowType(r);
            for (int j = 0; j < n; j++) {
                int o = (r * n + j) * w;
                System.arraycopy(b.values(), r * w, out.values(), o, w);
                System.arraycopy(b.mask(), r * w, out.mask(), o, w);
                Arrays.fill(out.group(), o, o + w, group * n + j);
                Arrays.fill(out.type(), o, o + w, type);
            }
        }
        return out;
    }

    /** {@code update_buffer_with_predictions}: each target trajectory's predictions written in as observations. */
    private static T0Batch update(T0Batch b, float[] prediction, int targets, int horizon, int n, int at) {
        T0Batch out = b.copy();
        int w = b.width(), rank = -1;
        for (int r = 0; r < b.rows(); r++) {
            if (b.rowType(r) != T0Batch.TARGET) {
                continue;
            }
            rank = Math.min(rank + 1, targets * n - 1);
            int t = rank / n, j = rank % n;
            for (int h = 0; h < horizon; h++) {
                out.values()[r * w + at + h] = prediction[(t * horizon + h) * n + j];
                out.mask()[r * w + at + h] = T0Batch.VALID;
            }
        }
        return out;
    }

    /**
     * {@code _predict_step}: one forward pass over a window whose last {@code horizon} steps are the forecast
     * region, {@code [rows · horizon · trained levels]} in data space.
     */
    float[] step(T0Batch window, int horizon) {
        int p = config.patchSize(), nt = config.quantiles(), w = window.width(), rows = window.rows();
        int ce = window.contextEnd();
        if (ce % p != 0 || (w - ce) % p != 0) {
            throw new IllegalStateException("forward window not aligned to whole patches");
        }
        T0Scaler.Scaled s = scaler.scale(window);
        T0Forward.Input in = T0Forward.Input.of(window, s.scaled(), p);
        int cp = (w - horizon) / p, count = horizon / p;
        float[] q = forward.forward(in, cp - 1, count);
        float[] out = new float[rows * horizon * nt];
        IntStream all = IntStream.range(0, rows);
        (rows > 1 && (long) rows * horizon * nt >= T0Scaler.PARALLEL_CELLS ? all.parallel() : all).forEach(r -> {
            for (int k = 0; k < count; k++) {
                int last = r * w + (cp - 1 + k) * p + p - 1;
                float loc = s.loc()[last], scale = s.scale()[last];
                for (int j = 0; j < p; j++) {
                    for (int i = 0; i < nt; i++) {
                        out[(r * horizon + k * p + j) * nt + i] = scaler.rescale(q[((r * count + k) * p + j) * nt + i], loc, scale);
                    }
                }
            }
        });
        if (observer != null) {
            observer.call(window, horizon, out);
        }
        return out;
    }

    private static float[] selectRows(float[] x, int[] rows, int width) {
        float[] out = new float[rows.length * width];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(x, rows[i] * width, out, i * width, width);
        }
        return out;
    }

    private static float[] cut(float[] x, int rows, int width, int keep, int inner) {
        float[] out = new float[rows * keep * inner];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(x, r * width * inner, out, r * keep * inner, keep * inner);
        }
        return out;
    }

    static int roundUp(int v, int m) {
        return (v + m - 1) / m * m;
    }

    private static double[] toDouble(float[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i];
        }
        return out;
    }

    private static boolean isClose(double a, double b) {
        return Math.abs(a - b) <= Math.max(1e-9 * Math.max(Math.abs(a), Math.abs(b)), 1e-9);
    }

    /** {@code _with_mirrors}: the levels plus each level's mirror {@code 1 − q}, ascending. */
    static double[] withMirrors(double[] levels) {
        List<Double> closed = new ArrayList<>();
        for (double l : levels) {
            closed.add(l);
        }
        for (double l : levels) {
            boolean found = false;
            for (double o : closed) {
                found |= isClose(1.0 - l, o);
            }
            if (!found) {
                closed.add(1.0 - l);
            }
        }
        return closed.stream().mapToDouble(Double::doubleValue).sorted().toArray();
    }

    private static int levelIndex(double[] levels, double level) {
        for (int i = 0; i < levels.length; i++) {
            if (isClose(level, levels[i])) {
                return i;
            }
        }
        throw new IllegalStateException("no level " + level);
    }

    /** Whether symmetric averaging changes a target: its history is not floored at zero. */
    private static boolean averaged(double[] history) {
        if (history.length == 0) {
            return false;
        }
        double min = Double.POSITIVE_INFINITY;
        for (double v : history) {
            min = Math.min(min, v);
        }
        return min != 0;
    }

    /** {@code _symmetric_average}: {@code (f(x)_q − f(−x)_{1−q}) / 2} at the requested levels. */
    private static double[] symmetricAverage(double[] positive, double[] negated, double[] levels, double[] requested,
            double[][] history, int horizon) {
        int nl = levels.length, nr = requested.length, targets = history.length;
        int[] mirror = new int[nl], pick = new int[nr];
        for (int i = 0; i < nl; i++) {
            mirror[i] = levelIndex(levels, 1.0 - levels[i]);
        }
        for (int i = 0; i < nr; i++) {
            pick[i] = levelIndex(levels, requested[i]);
        }
        double[] out = new double[targets * horizon * nr];
        for (int t = 0; t < targets; t++) {
            boolean avg = averaged(history[t]);
            for (int h = 0; h < horizon; h++) {
                int o = (t * horizon + h) * nl;
                for (int i = 0; i < nr; i++) {
                    int l = pick[i];
                    out[(t * horizon + h) * nr + i] = avg ? (positive[o + l] - negated[o + mirror[l]]) / 2 : positive[o + l];
                }
            }
        }
        return out;
    }

    /** {@code _bound_to_history}: clip non-negative histories at zero, round integer ones. */
    private void bound(double[] values, double[][] history, int horizon, int nq) {
        for (int t = 0; t < history.length; t++) {
            double[] seen = history[t];
            if (seen.length == 0) {
                continue;
            }
            double min = Double.POSITIVE_INFINITY;
            boolean integer = true;
            for (double v : seen) {
                min = Math.min(min, v);
                integer &= v == Math.rint(v);
            }
            for (int i = t * horizon * nq; i < (t + 1) * horizon * nq; i++) {
                if (options.clipNonnegative() && min >= 0) {
                    values[i] = Math.max(values[i], 0.0);
                }
                if (options.snapIntegers() && integer) {
                    values[i] = Math.rint(values[i]);
                }
            }
        }
    }
}
