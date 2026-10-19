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

/**
 * The host-side pieces of {@code tirex2.PostProcessor}: trend detection and first-order
 * differencing of the targets on the way in, and band calibration plus integration of differenced
 * rows on the way out. The constants are {@code PostProcessorConfig}'s defaults.
 */
public final class PostProcessor {

    static final double TREND_WINDOW = 1.0;
    static final double TREND_THRESHOLD = 1.5;
    static final double TREND_MIN_R2 = 0.87;
    static final double DIFF_BAND_EXPONENT = 0.51;
    static final double DIFF_BAND_SCALE = 1.68;
    static final double RAW_BAND_EXPONENT = 0.52;
    static final double RAW_BAND_SCALE = 0.90;

    private PostProcessor() {}

    /**
     * Whether a target row is trending: an OLS fit of the z-scored observed values against their
     * positions over the last {@code TREND_WINDOW} of the row, with {@code |slope|·length ≥ 1.5} and
     * {@code R² ≥ 0.87}. Rows with fewer than 4 observations never are.
     */
    public static boolean isTrending(float[] row) {
        int window = Math.max(1, (int) Math.rint(TREND_WINDOW * row.length));
        int start = row.length - window;
        int n = 0;
        for (int i = start; i < row.length; i++) {
            if (!Float.isNaN(row[i])) {
                n++;
            }
        }
        if (n < 4) {
            return false;
        }
        double[] idx = new double[n];
        double[] val = new double[n];
        int j = 0;
        for (int i = start; i < row.length; i++) {
            if (!Float.isNaN(row[i])) {
                idx[j] = i - start;
                val[j++] = row[i];
            }
        }
        double mean = 0;
        for (double v : val) {
            mean += v;
        }
        mean /= n;
        double var = 0;
        for (double v : val) {
            var += (v - mean) * (v - mean);
        }
        double std = Math.sqrt(var / (n - 1));
        if (std < 1e-12) {
            return false;
        }
        double[] z = new double[n];
        double zMean = 0, xMean = 0;
        for (int i = 0; i < n; i++) {
            z[i] = (val[i] - mean) / std;
            zMean += z[i];
            xMean += idx[i];
        }
        zMean /= n;
        xMean /= n;
        double sxy = 0, sxx = 0;
        for (int i = 0; i < n; i++) {
            sxy += (idx[i] - xMean) * (z[i] - zMean);
            sxx += (idx[i] - xMean) * (idx[i] - xMean);
        }
        double slope = sxy / sxx;
        double intercept = zMean - slope * xMean;
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++) {
            double r = z[i] - (slope * idx[i] + intercept);
            ssRes += r * r;
            ssTot += (z[i] - zMean) * (z[i] - zMean);
        }
        double r2 = ssTot > 1e-12 ? 1.0 - ssRes / ssTot : 0.0;
        double score = Math.abs(slope) * window;
        return score >= TREND_THRESHOLD && r2 >= TREND_MIN_R2;
    }

    /** {@code [0, x1−x0, x2−x1, …]} in float, as {@code _diff_forward}. */
    public static float[] diff(float[] row) {
        float[] d = new float[row.length];
        for (int i = 1; i < row.length; i++) {
            d[i] = row[i] - row[i - 1];
        }
        return d;
    }

    /** The last observed value of a row, or 0 if it has none ({@code _diff_params}). */
    public static float lastValue(float[] row) {
        for (int i = row.length - 1; i >= 0; i--) {
            if (!Float.isNaN(row[i])) {
                return row[i];
            }
        }
        return 0f;
    }

    /**
     * Calibrates one target row's quantiles {@code q [Q][H]} in place.
     *
     * <ul>
     *   <li>raw rows: {@code median + √0.90 · t^(0.52−0.5) · (q − median)};
     *   <li>differenced rows: {@code last + cumsum(median) + √1.68 · sign · √cumsum(offset²) · t^(0.51−0.5)},
     *       with {@code offset = q − median} and {@code sign = sign(Σ_t offset)} per quantile.
     * </ul>
     */
    public static void calibrate(float[][] q, boolean differenced, float last) {
        int nq = q.length;
        int h = q[0].length;
        float[] median = q[nq / 2].clone();
        if (!differenced) {
            double s = Math.sqrt(RAW_BAND_SCALE);
            for (int k = 0; k < nq; k++) {
                for (int t = 0; t < h; t++) {
                    double factor = Math.pow(t + 1, RAW_BAND_EXPONENT - 0.5);
                    q[k][t] = (float) (median[t] + s * factor * (q[k][t] - median[t]));
                }
            }
            return;
        }
        double s = Math.sqrt(DIFF_BAND_SCALE);
        double[] path = new double[h];
        double acc = 0;
        for (int t = 0; t < h; t++) {
            acc += median[t];
            path[t] = acc;
        }
        for (int k = 0; k < nq; k++) {
            double sum = 0;
            for (int t = 0; t < h; t++) {
                sum += q[k][t] - median[t];
            }
            double sign = Math.signum(sum);
            double ss = 0;
            for (int t = 0; t < h; t++) {
                double off = q[k][t] - median[t];
                ss += off * off;
                double band = sign * Math.sqrt(ss) * Math.pow(t + 1, DIFF_BAND_EXPONENT - 0.5);
                q[k][t] = (float) (path[t] + s * band + last);
            }
        }
    }
}
