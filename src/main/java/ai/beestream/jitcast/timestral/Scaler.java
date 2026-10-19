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

/**
 * {@code timestral.scaler}: causal per-patch mean/std normalization with an asinh squash, the
 * whole-row statistics of known-future covariates, and the advancing refine frames of a decoded
 * block. Statistics run in double and are rounded to float where the reference rounds them, so the
 * engines see the reference's values bit for bit or within one rounding.
 */
public final class Scaler {

    /** {@code torch.finfo(torch.float32).eps}, the floor of the divisor in the affine transform. */
    static final float FLOAT32_EPS = Math.ulp(1.0f);

    private Scaler() {}

    /**
     * Per-patch {@code (loc, scale)} of one row, each broadcast to every step of its patch: the
     * causal mean and std (Bessel-corrected, floored at {@code minimumScale}) of every observed value
     * up to the end of the patch ({@code causal_patch_loc_scale}).
     */
    public static void causalPatchLocScale(float[] x, boolean[] observed, int patch, float minimumScale, float[] loc,
            float[] scale) {
        int t = x.length;
        double cumW = 0, cumX = 0, m2 = 0, prevMean = 0;
        for (int i = 0; i < t; i++) {
            double w = observed[i] ? 1.0 : 0.0;
            double xi = x[i];
            cumW += w;
            cumX += w * xi;
            double mean = cumX / Math.max(cumW, 1.0);
            m2 += (xi - prevMean) * (xi - mean) * w;
            prevMean = mean;
            if (i % patch == patch - 1) {
                double var = m2 / Math.max(Math.max(cumW, 1.0) - 1.0, 1.0);
                float l = (float) mean;
                float s = (float) Math.max(Math.sqrt(var), minimumScale);
                for (int j = i - patch + 1; j <= i; j++) {
                    loc[j] = l;
                    scale[j] = s;
                }
            }
        }
    }

    /**
     * The {@code (loc, scale)} of a row's last patch, as {@link #causalPatchLocScale} computes it at
     * the row's last step, in one pass and without the per-step arrays; the row is a whole number of
     * patches long.
     */
    public static float[] lastLocScale(float[] x, boolean[] observed, float minimumScale) {
        double cumW = 0, cumX = 0, m2 = 0, prevMean = 0, mean = 0;
        for (int i = 0; i < x.length; i++) {
            double w = observed[i] ? 1.0 : 0.0;
            double xi = x[i];
            cumW += w;
            cumX += w * xi;
            mean = cumX / Math.max(cumW, 1.0);
            m2 += (xi - prevMean) * (xi - mean) * w;
            prevMean = mean;
        }
        double var = m2 / Math.max(Math.max(cumW, 1.0) - 1.0, 1.0);
        return new float[] {(float) mean, (float) Math.max(Math.sqrt(var), minimumScale)};
    }

    /** {@code asinh((x − loc) / scale)} on observed positions, 0 elsewhere ({@code apply_affine_asinh_masked}). */
    public static float affine(float x, float loc, float scale, boolean observed, boolean asinh) {
        if (!observed) {
            return 0f;
        }
        float v = (x - loc) / Math.max(scale, FLOAT32_EPS);
        return asinh ? asinh(v) : v;
    }

    public static float asinh(float v) {
        double a = Math.abs((double) v);
        double r = Math.log(a + Math.sqrt(a * a + 1.0));
        return (float) (v < 0 ? -r : r);
    }

    /**
     * Whole-row statistics of a known-future covariate ({@code instance_norm_stats}), in float as the
     * reference computes them: {@code {loc, scale, binary}}. A row whose observed values are all 0 or
     * 1 is an indicator and keeps {@code loc = 0, scale = 1} (and skips the asinh).
     */
    public static float[] instanceNorm(float[] x, boolean[] observed, float minimumScale) {
        float n = 0, sum = 0;
        boolean zeroOne = true, any = false;
        for (int i = 0; i < x.length; i++) {
            if (observed[i]) {
                n += 1f;
                sum += x[i];
                any = true;
            }
            zeroOne &= x[i] == 0f || x[i] == 1f || !observed[i];
        }
        n = Math.max(n, 1f);
        float loc = sum / n;
        float ss = 0;
        for (int i = 0; i < x.length; i++) {
            float d = observed[i] ? x[i] - loc : 0f;
            ss += d * d;
        }
        float scale = Math.max((float) Math.sqrt(ss / n), minimumScale);
        boolean binary = zeroOne && any;
        return binary ? new float[] {0f, 1f, 1f} : new float[] {loc, scale, 0f};
    }

    /** {@code sinh(x)·scale + loc} ({@code invert_affine_asinh}). */
    public static float invert(float x, float loc, float scale, boolean asinh) {
        float v = asinh ? (float) Math.sinh(x) : x;
        return v * scale + loc;
    }

    /**
     * Maps ±inf (a sinh overflow) to the largest / smallest finite value of the vector, and a vector
     * with no finite value to 0, in place ({@code clamp_nonfinite} along the quantiles).
     */
    public static void clampNonFinite(float[] v) {
        float max = Float.NEGATIVE_INFINITY, min = Float.POSITIVE_INFINITY;
        for (float x : v) {
            if (Float.isFinite(x)) {
                max = Math.max(max, x);
                min = Math.min(min, x);
            }
        }
        for (int i = 0; i < v.length; i++) {
            float x = v[i] == Float.POSITIVE_INFINITY ? max : v[i] == Float.NEGATIVE_INFINITY ? min : v[i];
            v[i] = Float.isInfinite(x) ? 0f : x;
        }
    }

    /**
     * The per-patch {@code (loc, scale)} of a decoded block in the advancing refine frame
     * ({@code advancing_block_frames}): horizon patch {@code j} uses the statistics of the observed
     * context plus the block's medians of patches {@code 0 .. j−1}, each folded in after being
     * denormalized in its own frame. In double, as Welford states merged in the reference's order.
     *
     * @param context one row's block context, a multiple of the patch size long
     * @param sinhMedian {@code [H][P]} the block's median in the last context patch's frame with the
     *     asinh undone, {@code (median − loc) / scale}
     * @return {@code {loc[H], scale[H]}}
     */
    public static double[][] advancingFrames(float[] context, boolean[] observed, double[][] sinhMedian, int patch,
            float minimumScale) {
        int nPred = sinhMedian.length;
        int nCtx = context.length / patch;
        int n = nCtx + nPred;
        double[] w = new double[n], mean = new double[n], m2 = new double[n];
        for (int p = 0; p < nCtx; p++) {
            double cnt = 0, sum = 0;
            for (int i = p * patch; i < (p + 1) * patch; i++) {
                double wi = observed[i] ? 1.0 : 0.0;
                cnt += wi;
                sum += context[i] * wi;
            }
            double mu = sum / Math.max(cnt, 1.0);
            double s2 = 0;
            for (int i = p * patch; i < (p + 1) * patch; i++) {
                double wi = observed[i] ? 1.0 : 0.0;
                double dv = context[i] - mu;
                s2 += wi * (dv * dv);
            }
            w[p] = cnt;
            mean[p] = cnt == 0 ? 0 : mu;
            m2[p] = cnt == 0 ? 0 : s2;
        }
        // cumulative_welford: an inclusive Hillis-Steele scan, merged in the reference's order
        for (int d = 1; d < n; d *= 2) {
            double[] w2 = new double[n], mean2 = new double[n], m22 = new double[n];
            for (int i = 0; i < n; i++) {
                double wa = i >= d ? w[i - d] : 0, ma = i >= d ? mean[i - d] : 0, m2a = i >= d ? m2[i - d] : 0;
                double[] c = combine(wa, ma, m2a, w[i], mean[i], m2[i]);
                w2[i] = c[0];
                mean2[i] = c[1];
                m22[i] = c[2];
            }
            w = w2;
            mean = mean2;
            m2 = m22;
        }
        double cnt = w[n - nPred - 1], mu = mean[n - nPred - 1], mm = m2[n - nPred - 1];
        double offset = mu;
        double clip = Math.sqrt(Float.MAX_VALUE) / 1e3;
        double[] locs = new double[nPred], scales = new double[nPred];
        for (int h = 0; h < nPred; h++) {
            double var = Math.max(mm / Math.max(cnt - 1.0, 1.0), 0.0);
            double s = Math.max(Math.sqrt(var), minimumScale);
            locs[h] = mu;
            scales[h] = s;
            double[] folded = new double[patch];
            double sum = 0;
            for (int i = 0; i < patch; i++) {
                double f = Math.max(-clip, Math.min(clip, sinhMedian[h][i] * s + mu - offset));
                folded[i] = f + offset;
                sum += folded[i];
            }
            double bMean = sum / patch;
            double bM2 = 0;
            for (int i = 0; i < patch; i++) {
                double dv = folded[i] - bMean;
                bM2 += dv * dv;
            }
            double[] c = combine(cnt, mu, mm, patch, bMean, bM2);
            cnt = c[0];
            mu = c[1];
            mm = c[2];
        }
        return new double[][] {locs, scales};
    }

    /** {@code welford_combine}: merges two {@code (weight, mean, m2)} states; {@code (0, 0, 0)} is the identity. */
    static double[] combine(double wa, double ma, double m2a, double wb, double mb, double m2b) {
        double w = wa + wb;
        double wSafe = Math.max(w, Double.MIN_NORMAL);
        double delta = mb - ma;
        double mean = ma + delta * wb / wSafe;
        double m2 = m2a + m2b + delta * delta * wa * wb / wSafe;
        return w == 0 ? new double[] {w, 0, 0} : new double[] {w, mean, m2};
    }
}
