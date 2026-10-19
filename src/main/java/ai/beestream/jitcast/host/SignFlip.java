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
package ai.beestream.jitcast.host;

import ai.beestream.jitcast.api.TimeSeries;

/**
 * Sign-flip test-time augmentation: forecast the series and its negation, then average the first
 * forecast with the negated mirror of the second, matching each quantile level with its complement.
 * It needs a quantile set symmetric around 0.5.
 */
public final class SignFlip {

    private SignFlip() {}

    /** The series with every target and covariate value negated. */
    public static TimeSeries negate(TimeSeries ts) {
        return new TimeSeries(neg(ts.target()), neg(ts.pastCovariates()), neg(ts.futureCovariates()));
    }

    private static float[][] neg(float[][] m) {
        if (m == null) {
            return null;
        }
        float[][] r = new float[m.length][];
        for (int i = 0; i < m.length; i++) {
            r[i] = new float[m[i].length];
            for (int j = 0; j < m[i].length; j++) {
                r[i][j] = -m[i][j];
            }
        }
        return r;
    }

    /** {@code complement[i]} = the level closest to {@code 1 − q[i]}; fails unless within 1e-6. */
    public static int[] complement(float[] q) {
        int[] c = new int[q.length];
        for (int i = 0; i < q.length; i++) {
            int best = 0;
            for (int j = 1; j < q.length; j++) {
                if (Math.abs(q[j] - (1 - q[i])) < Math.abs(q[best] - (1 - q[i]))) {
                    best = j;
                }
            }
            if (Math.abs(q[best] - (1 - q[i])) > 1e-6) {
                throw new IllegalStateException("sign-flip TTA needs a symmetric quantile set");
            }
            c[i] = best;
        }
        return c;
    }

    /**
     * Averages one series' forecast {@code a [targets][quantiles][≥ h]} with the forecast of its
     * negation {@code b}, in place over the first {@code h} steps: {@code a[q] = (a[q] − b[c[q]]) / 2}.
     */
    public static void average(float[][][] a, float[][][] b, int[] complement, int h) {
        for (int t = 0; t < a.length; t++) {
            for (int q = 0; q < a[t].length; q++) {
                for (int i = 0; i < h; i++) {
                    a[t][q][i] = (a[t][q][i] + -b[t][complement[q]][i]) / 2;
                }
            }
        }
    }
}
