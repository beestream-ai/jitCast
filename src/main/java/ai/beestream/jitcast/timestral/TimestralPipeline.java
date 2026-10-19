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
package ai.beestream.jitcast.timestral;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.host.SeriesWindows;
import ai.beestream.jitcast.host.SignFlip;

/**
 * {@code TimestralForecaster.forecast()} around any {@link BlockForward}: windows of at most
 * {@code batchSize} series and {@code maxTokens} tokens, batch assembly (one row per target and
 * covariate, right-aligned, capped at {@code maxContext} and left-padded to whole patches),
 * block-autoregressive decoding with each block moved to the advancing refine frame and its median
 * fed back as context, the cap to the observed range, and the clampsym test-time augmentation
 * (sign-flip averaging and clipping non-negative series at 0).
 *
 * <p>A series' forecast does not depend on the other series of its batch: rows are right-aligned,
 * so patch boundaries count from the forecast start, and time positions from each series' first
 * observation. Categorical covariates are target-encoded first ({@link TargetEncoding}).
 */
public final class TimestralPipeline {

    /** The default batch size, as {@code jitcast}'s other pipelines. */
    public static final int DEFAULT_BATCH_SIZE = 512;

    /** The most tokens one forward pass takes ({@code -Dtimestral.maxTokens}, default 131,072). */
    public static final long DEFAULT_MAX_TOKENS = Long.getLong("timestral.maxTokens", 131_072L);

    /**
     * The forecaster's settings, with the reference's defaults.
     *
     * @param maxContext context cap in time steps (the most recent are kept)
     * @param perShotHorizonPatches patches decoded per block, capped at the trained block size
     * @param capK forecasts are clamped to {@code [min − k·s, max + k·s]} of the observed context,
     *     {@code s} the scale of its last patch; 0 disables
     * @param extremeValueThreshold fed-back medians at or above this magnitude become 0
     * @param symmetricAveraging average with the reflected forecast of the negated series
     * @param clipNonnegative clamp at 0 the series whose observed context is non-negative
     */
    public record Options(int maxContext, int perShotHorizonPatches, float capK, float extremeValueThreshold,
            boolean symmetricAveraging, boolean clipNonnegative) {

        public static Options defaults(TimestralConfig c) {
            return new Options(c.contextLength(), 32, 1e4f, 1e30f, true, true);
        }

        public Options withSymmetricAveraging(boolean on) {
            return new Options(maxContext, perShotHorizonPatches, capK, extremeValueThreshold, on, clipNonnegative);
        }

        public Options withClipNonnegative(boolean on) {
            return new Options(maxContext, perShotHorizonPatches, capK, extremeValueThreshold, symmetricAveraging, on);
        }
    }

    private final TimestralConfig config;
    private final BlockForward forward;
    private final Options options;
    private final int batchSize;
    private final long maxTokens;
    private final int[] complement;

    public TimestralPipeline(TimestralConfig config, BlockForward forward) {
        this(config, forward, Options.defaults(config), DEFAULT_BATCH_SIZE, DEFAULT_MAX_TOKENS);
    }

    public TimestralPipeline(TimestralConfig config, BlockForward forward, Options options, int batchSize, long maxTokens) {
        if (batchSize < 1 || maxTokens < 1) {
            throw new IllegalArgumentException("batchSize and maxTokens must be >= 1");
        }
        if (options.maxContext() < config.patchSize() || options.perShotHorizonPatches() < 1) {
            throw new IllegalArgumentException("maxContext must hold a patch and perShotHorizonPatches must be >= 1");
        }
        this.config = config;
        this.forward = forward;
        this.options = options;
        this.batchSize = batchSize;
        this.maxTokens = maxTokens;
        this.complement = options.symmetricAveraging() ? SignFlip.complement(config.quantileLevels()) : null;
    }

    public Options options() {
        return options;
    }

    public TimestralPipeline withOptions(Options o) {
        return new TimestralPipeline(config, forward, o, batchSize, maxTokens);
    }

    public TimestralPipeline withMaxTokens(long tokens) {
        return new TimestralPipeline(config, forward, options, batchSize, tokens);
    }

    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        if (horizon < 1) {
            throw new IllegalArgumentException("horizon must be >= 1, got " + horizon);
        }
        int ps = config.patchSize();
        int maxLen = series.stream().mapToInt(TimeSeries::length).max().orElse(0);
        int perShot = Math.min(options.perShotHorizonPatches(), config.blockSize());
        int patches = (Math.min(maxLen, options.maxContext()) + ps - 1) / ps + 2 * Math.min(perShot, (horizon + ps - 1) / ps);
        List<Forecast> out = new ArrayList<>(series.size());
        for (int[] w : SeriesWindows.split(series, batchSize, maxTokens, patches)) {
            out.addAll(forecastWindow(series.subList(w[0], w[1]), horizon));
        }
        return out;
    }

    private List<Forecast> forecastWindow(List<TimeSeries> series, int horizon) {
        for (TimeSeries ts : series) {
            if (ts.rows() > ts.targets() && !config.covariateEnabled()) {
                throw new IllegalArgumentException("this checkpoint was not trained with covariates");
            }
            if (ts.futureCount() + ts.futureCategoricalCount() > 0 && !config.supportsKnownCovariates()) {
                throw new IllegalArgumentException("this checkpoint cannot condition on known-future covariates");
            }
        }
        // categorical covariates become numeric ones, encoded once against the series' own target
        List<TimeSeries> encoded = series.stream().map(ts -> TargetEncoding.apply(ts, horizon, options.maxContext())).toList();
        List<float[][][]> a = forecastRaw(encoded, horizon, 1f);
        if (options.symmetricAveraging()) {
            List<float[][][]> b = forecastRaw(encoded, horizon, -1f);
            for (int s = 0; s < a.size(); s++) {
                SignFlip.average(a.get(s), b.get(s), complement, horizon);
            }
        }
        List<Forecast> out = new ArrayList<>(series.size());
        for (int s = 0; s < series.size(); s++) {
            float[][][] f = a.get(s);
            if (options.clipNonnegative()) {
                for (int v = 0; v < f.length; v++) {
                    if (nonNegative(series.get(s).target()[v])) {
                        for (float[] level : f[v]) {
                            for (int t = 0; t < level.length; t++) {
                                level[t] = Math.max(level[t], 0f);
                            }
                        }
                    }
                }
            }
            out.add(new Forecast(config.quantileLevels(), f));
        }
        return out;
    }

    /** Whether every observed (finite) context value is non-negative, and one exists. */
    static boolean nonNegative(float[] row) {
        boolean any = false;
        for (float v : row) {
            if (Float.isFinite(v)) {
                if (v < 0) {
                    return false;
                }
                any = true;
            }
        }
        return any;
    }

    /** The packed rows of one window, as {@code _assemble_rows} builds them. */
    record Assembled(float[][] rows, boolean[][] mask, int[] group, int[] role, float[][] knownFuture,
            boolean[][] knownFutureMask, int[][] targetRows) {}

    /** {@code _assemble_rows} for a window, with every value multiplied by {@code sign}. */
    Assembled assemble(List<TimeSeries> series, int horizon, float sign) {
        int ps = config.patchSize();
        int t = series.stream().mapToInt(TimeSeries::length).max().orElse(0);
        int horizonLen = (horizon + ps - 1) / ps * ps;
        boolean covariates = series.stream().anyMatch(s -> s.pastCount() > 0 || s.futureCount() > 0);
        boolean anyKnown = series.stream().anyMatch(s -> s.futureCount() > 0);
        List<float[]> rows = new ArrayList<>();
        List<boolean[]> masks = new ArrayList<>();
        List<Integer> groups = new ArrayList<>(), roles = new ArrayList<>();
        List<float[]> kf = new ArrayList<>();
        List<boolean[]> kfm = new ArrayList<>();
        int[][] targetRows = new int[series.size()][];
        for (int s = 0; s < series.size(); s++) {
            TimeSeries ts = series.get(s);
            int len = ts.length();
            targetRows[s] = new int[ts.targets()];
            for (int v = 0; v < ts.targets(); v++) {
                targetRows[s][v] = rows.size();
                float[] x = new float[t];
                boolean[] m = new boolean[t];
                for (int i = 0; i < len; i++) {
                    float val = ts.target()[v][i];
                    m[t - len + i] = Float.isFinite(val);
                    // the reference zeroes missing targets and then negates them: -0.0 in the mirrored pass
                    x[t - len + i] = sign * (m[t - len + i] ? val : 0f);
                }
                rows.add(forwardFill(x, m));
                masks.add(m);
                groups.add(s);
                roles.add(BlockInput.TARGET);
                kf.add(new float[horizonLen]);
                kfm.add(new boolean[horizonLen]);
            }
            for (int v = 0; v < ts.pastCount(); v++) {
                addCovariate(ts.pastCovariates()[v], len, t, sign, rows, masks);
                groups.add(s);
                roles.add(BlockInput.PAST);
                kf.add(new float[horizonLen]);
                kfm.add(new boolean[horizonLen]);
            }
            for (int v = 0; v < ts.futureCount(); v++) {
                float[] fc = ts.futureCovariates()[v];
                if (fc.length < len + horizon) {
                    throw new IllegalArgumentException("future covariates of series " + s + " need at least "
                            + (len + horizon) + " steps, got " + fc.length);
                }
                addCovariate(fc, len, t, sign, rows, masks);
                groups.add(s);
                roles.add(BlockInput.KNOWN);
                float[] f = new float[horizonLen];
                boolean[] fm = new boolean[horizonLen];
                for (int i = 0; i < horizon; i++) {
                    float val = fc[len + i];
                    fm[i] = !Float.isNaN(val);
                    f[i] = fm[i] ? sign * val : 0f;
                }
                kf.add(f);
                kfm.add(fm);
            }
        }
        // the most recent maxContext steps, left-padded to whole patches
        int kept = Math.min(t, options.maxContext());
        int padded = (kept + ps - 1) / ps * ps;
        float[][] r = new float[rows.size()][padded];
        boolean[][] m = new boolean[rows.size()][padded];
        for (int i = 0; i < r.length; i++) {
            System.arraycopy(rows.get(i), t - kept, r[i], padded - kept, kept);
            System.arraycopy(masks.get(i), t - kept, m[i], padded - kept, kept);
        }
        return new Assembled(r, m, groups.stream().mapToInt(Integer::intValue).toArray(),
                covariates ? roles.stream().mapToInt(Integer::intValue).toArray() : null,
                anyKnown ? kf.toArray(float[][]::new) : null, anyKnown ? kfm.toArray(boolean[][]::new) : null, targetRows);
    }

    private static void addCovariate(float[] values, int len, int t, float sign, List<float[]> rows, List<boolean[]> masks) {
        float[] x = new float[t];
        boolean[] m = new boolean[t];
        for (int i = 0; i < len; i++) {
            float val = values[i];
            m[t - len + i] = !Float.isNaN(val);
            x[t - len + i] = m[t - len + i] ? sign * val : 0f;
        }
        rows.add(x);
        masks.add(m);
    }

    /** {@code ffill_with_mask}: unobserved positions take the last observed value; leading ones stay 0. */
    static float[] forwardFill(float[] x, boolean[] m) {
        float[] out = x.clone();
        float last = 0f;
        boolean seen = false;
        for (int i = 0; i < x.length; i++) {
            if (m[i]) {
                last = x[i];
                seen = true;
            } else {
                out[i] = seen ? last : x[0];
            }
        }
        return out;
    }

    /** One forecast pass ({@code _forecast_raw}): per series, {@code [targets][quantiles][horizon]}. */
    private List<float[][][]> forecastRaw(List<TimeSeries> series, int horizon, float sign) {
        long tm = Timing.start();
        Assembled a = assemble(series, horizon, sign);
        Timing.add("host.assemble", tm);
        float[][][] q = decode(a, horizon);
        tm = Timing.start();
        int nq = config.numQuantiles();
        float[][] bounds = new float[a.rows().length][];
        IntStream.range(0, series.size()).parallel().forEach(s -> {
            for (int row : a.targetRows()[s]) {
                bounds[row] = capBounds(a.rows()[row], a.mask()[row]);
            }
        });
        List<float[][][]> out = new ArrayList<>(series.size());
        for (int s = 0; s < series.size(); s++) {
            int[] tr = a.targetRows()[s];
            float[][][] f = new float[tr.length][nq][horizon];
            for (int v = 0; v < tr.length; v++) {
                int row = tr[v];
                float[] b = bounds[row];
                for (int t = 0; t < horizon; t++) {
                    for (int k = 0; k < nq; k++) {
                        float x = q[row][t][k];
                        if (b != null) {
                            x = Math.min(Math.max(x, b[0]), b[1]);
                        }
                        if (Float.isNaN(x)) {
                            throw new IllegalStateException("forecast produced NaN values; the input is degenerate");
                        }
                        f[v][k][t] = x;
                    }
                }
            }
            out.add(f);
        }
        Timing.add("host.cap", tm);
        return out;
    }

    /** {@code _cap_quantiles}: {@code [min − k·s, max + k·s]} of a row's observed context, or null when disabled. */
    private float[] capBounds(float[] row, boolean[] mask) {
        if (options.capK() <= 0) {
            return null;
        }
        float max = Float.NEGATIVE_INFINITY, min = Float.POSITIVE_INFINITY;
        for (int i = 0; i < row.length; i++) {
            if (mask[i]) {
                max = Math.max(max, row[i]);
                min = Math.min(min, row[i]);
            }
        }
        max = max == Float.NEGATIVE_INFINITY ? 0f : max;
        min = min == Float.POSITIVE_INFINITY ? 0f : min;
        float[] loc = new float[row.length], scale = new float[row.length];
        Scaler.causalPatchLocScale(row, mask, config.patchSize(), config.minimumScale(), loc, scale);
        float s = scale[row.length - 1] * options.capK();
        return new float[] {min - s, max + s};
    }

    /**
     * {@code _block_ar_decode}: block after block, the context growing by each block's feedback
     * (targets their median, known covariates their real future, past covariates a blank), capped
     * at {@code maxContext}. Returns {@code [rows][horizon][quantiles]}.
     */
    private float[][][] decode(Assembled a, int horizon) {
        int ps = config.patchSize();
        int r = a.rows().length;
        int perShot = Math.min(options.perShotHorizonPatches(), config.blockSize());
        int median = config.medianIndex();
        float[][] context = a.rows();
        boolean[][] mask = a.mask();
        float[][][] out = new float[r][horizon][];
        int remaining = (horizon + ps - 1) / ps;
        int offset = 0;
        while (remaining > 0) {
            int block = Math.min(perShot, remaining);
            int blen = block * ps;
            float[][] kf = slice(a.knownFuture(), offset, blen);
            boolean[][] kfm = slice(a.knownFutureMask(), offset, blen);
            float[][][] q = forward.predictBlock(new BlockInput(context, mask, a.group(), block, a.role(), kf, kfm));
            long tm = Timing.start();
            refine(q, context, mask, a.role());
            Timing.add("host.refine", tm);
            for (int i = 0; i < r; i++) {
                for (int t = 0; t < blen && offset + t < horizon; t++) {
                    out[i][offset + t] = q[i][t];
                }
            }
            remaining -= block;
            offset += blen;
            if (remaining <= 0) {
                break;
            }
            int len = context[0].length + blen;
            int keep = len > options.maxContext() ? Math.max(ps, options.maxContext() / ps * ps) : len;
            float[][] next = new float[r][keep];
            boolean[][] nextMask = new boolean[r][keep];
            for (int i = 0; i < r; i++) {
                float[] row = new float[len];
                boolean[] m = new boolean[len];
                System.arraycopy(context[i], 0, row, 0, context[i].length);
                System.arraycopy(mask[i], 0, m, 0, mask[i].length);
                int role = a.role() == null ? BlockInput.TARGET : a.role()[i];
                for (int t = 0; t < blen; t++) {
                    int o = context[i].length + t;
                    if (role == BlockInput.KNOWN) {
                        row[o] = kf[i][t];
                        m[o] = kfm[i][t];
                    } else if (role == BlockInput.PAST) {
                        row[o] = 0f;
                        m[o] = false;
                    } else {
                        float v = q[i][t][median];
                        row[o] = Float.isFinite(v) && Math.abs(v) < options.extremeValueThreshold() ? v : 0f;
                        m[o] = true;
                    }
                }
                System.arraycopy(row, len - keep, next[i], 0, keep);
                System.arraycopy(m, len - keep, nextMask[i], 0, keep);
            }
            context = next;
            mask = nextMask;
        }
        return out;
    }

    /**
     * {@code _refine_block}: moves a decoded block of the target rows from the frame of the last
     * context patch to the advancing refine frame it was trained in; the first horizon patch keeps
     * its frame.
     */
    private void refine(float[][][] q, float[][] context, boolean[][] mask, int[] role) {
        int ps = config.patchSize();
        int nPred = q[0].length / ps;
        if (nPred <= 1) {
            return;
        }
        int median = config.medianIndex();
        int t = context[0].length;
        IntStream.range(0, q.length).parallel().forEach(i -> {
            if (role != null && role[i] != BlockInput.TARGET) {
                return;
            }
            float[] loc = new float[t], scale = new float[t];
            Scaler.causalPatchLocScale(context[i], mask[i], ps, config.minimumScale(), loc, scale);
            float loc0 = loc[t - 1], scale0 = scale[t - 1];
            double[][] sinhMedian = new double[nPred][ps];
            for (int j = 0; j < nPred; j++) {
                for (int s = 0; s < ps; s++) {
                    sinhMedian[j][s] = ((double) q[i][j * ps + s][median] - loc0) / scale0;
                }
            }
            double[][] frames = Scaler.advancingFrames(context[i], mask[i], sinhMedian, ps, config.minimumScale());
            for (int j = 0; j < nPred; j++) {
                float lj = j == 0 ? loc0 : (float) frames[0][j];
                float sj = j == 0 ? scale0 : (float) frames[1][j];
                for (int s = 0; s < ps; s++) {
                    float[] v = q[i][j * ps + s];
                    for (int k = 0; k < v.length; k++) {
                        v[k] = (v[k] - loc0) / scale0 * sj + lj;
                    }
                    Scaler.clampNonFinite(v);
                }
            }
        });
    }

    private static float[][] slice(float[][] m, int from, int len) {
        if (m == null) {
            return null;
        }
        float[][] r = new float[m.length][len];
        for (int i = 0; i < m.length; i++) {
            System.arraycopy(m[i], from, r[i], 0, len);
        }
        return r;
    }

    private static boolean[][] slice(boolean[][] m, int from, int len) {
        if (m == null) {
            return null;
        }
        boolean[][] r = new boolean[m.length][len];
        for (int i = 0; i < m.length; i++) {
            System.arraycopy(m[i], from, r[i], 0, len);
        }
        return r;
    }
}
