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

import java.util.Arrays;

/**
 * {@code chronos.utils}: linear interpolation between quantile levels and quantiles of a weighted
 * sample, as the long-horizon unrolling uses them, in float as the reference computes them.
 */
public final class QuantileMath {

    private QuantileMath() {}

    /**
     * {@code _get_prob_mass_per_quantile_level}: each level's share of the probability,
     * {@code (q[i+1] − q[i−1]) / 2} with {@code q[−1] = 0} and {@code q[n] = 1}, normalized.
     */
    public static float[] probabilityMass(float[] levels) {
        int n = levels.length;
        float[] b = new float[n + 2];
        System.arraycopy(levels, 0, b, 1, n);
        b[n + 1] = 1f;
        float[] m = new float[n];
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            m[i] = (b[i + 2] - b[i]) / 2f;
            sum += m[i];
        }
        for (int i = 0; i < n; i++) {
            m[i] = m[i] / sum;
        }
        return m;
    }

    /**
     * {@code interpolate_quantiles} over a batch: row {@code r}'s values at the {@code query} levels,
     * from its {@code values[r]} at {@code levels[r]} (each row's levels may differ). Below the lowest
     * level and above the highest the end values extend to 0 and 1; as in the reference, whether
     * those end points are added is decided once for the whole batch.
     */
    public static float[][] interpolate(float[] query, float[][] levels, float[][] values) {
        int rows = values.length;
        float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
        for (float[] l : levels) {
            for (float v : l) {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        }
        boolean padLow = min > 0f, padHigh = max < 1f;
        float[][] out = new float[rows][query.length];
        for (int r = 0; r < rows; r++) {
            int n = levels[r].length;
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            float[] lr = levels[r];
            Arrays.sort(order, (a, b) -> Float.compare(lr[a], lr[b]));
            int m = n + (padLow ? 1 : 0) + (padHigh ? 1 : 0);
            float[] sl = new float[m], sv = new float[m];
            int k = 0;
            if (padLow) {
                sl[k] = 0f;
                sv[k++] = values[r][order[0]];
            }
            for (int i = 0; i < n; i++) {
                sl[k] = lr[order[i]];
                sv[k++] = values[r][order[i]];
            }
            if (padHigh) {
                sl[k] = 1f;
                sv[k] = values[r][order[n - 1]];
            }
            for (int j = 0; j < query.length; j++) {
                float q = query[j];
                // torch.searchsorted(right=True): the first level above q
                int upper = 0;
                while (upper < m && sl[upper] <= q) {
                    upper++;
                }
                upper = Math.min(upper, m - 1);
                int lower = Math.max(upper - 1, 0);
                float w = nanToNum((q - sl[lower]) / (sl[upper] - sl[lower]));
                out[r][j] = sv[lower] + w * (sv[upper] - sv[lower]);
            }
        }
        return out;
    }

    /**
     * {@code weighted_quantile}: the {@code query} levels of each row of {@code samples}, every
     * sample {@code s} carrying probability {@code weights[s]} (normalized here): the samples are
     * sorted, their cumulative weights become their levels, and the levels are interpolated.
     */
    public static float[][] weightedQuantile(float[] query, float[] weights, float[][] samples) {
        int rows = samples.length, n = weights.length;
        float sum = 0f;
        for (float w : weights) {
            sum += w;
        }
        float[] norm = new float[n];
        for (int i = 0; i < n; i++) {
            norm[i] = weights[i] / sum;
        }
        float[][] levels = new float[rows][n];
        float[][] sorted = new float[rows][n];
        for (int r = 0; r < rows; r++) {
            float[] s = samples[r];
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Float.compare(s[a], s[b]));
            double acc = 0;
            for (int i = 0; i < n; i++) {
                sorted[r][i] = s[order[i]];
                // torch.cumsum on the CPU accumulates float in double
                acc += norm[order[i]];
                levels[r][i] = Math.min(Math.max((float) acc, 0f), 1f);
            }
        }
        return interpolate(query, levels, sorted);
    }

    /** {@code torch.nan_to_num}: NaN to 0, ±∞ to the largest finite floats. */
    private static float nanToNum(float v) {
        if (Float.isNaN(v)) {
            return 0f;
        }
        if (v == Float.POSITIVE_INFINITY) {
            return Float.MAX_VALUE;
        }
        return v == Float.NEGATIVE_INFINITY ? -Float.MAX_VALUE : v;
    }
}
