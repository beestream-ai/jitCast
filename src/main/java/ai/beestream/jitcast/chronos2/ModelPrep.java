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

import java.util.stream.IntStream;

/**
 * The host ends of {@code Chronos2Model.forward}, shared by every engine: the instance norm and the
 * patch tokens in front of the encoder, and the inverse norm after the output patch embedding. They
 * are {@code O(rows · length)} and run with the rows in parallel (each row's arithmetic is
 * unchanged, so the results do not depend on the thread count).
 *
 * @param tokens {@code [rows · (contextPatches + outputPatches) · 3P]}: the input patch embedding's
 *     input, each row's context patches then its future patches, each as its time encoding, its
 *     scaled values (0 where missing) and its observation mask
 * @param loc {@code [rows]} each row's location
 * @param scale {@code [rows]} each row's scale
 */
public record ModelPrep(float[] tokens, float[] loc, float[] scale) {

    /** {@code InstanceNorm}'s floor for a constant row. */
    static final float EPS = 1e-5f;

    public static ModelPrep of(Chronos2Config c, ModelLayout lay) {
        int r = lay.rows(), ps = c.patchSize(), nc = lay.contextPatches(), h = lay.outputPatches();
        int ne = nc + h, w = 3 * ps;
        float[] tok = new float[r * ne * w];
        float[] loc = new float[r], scale = new float[r];
        float timeScale = c.timeEncodingScale();
        IntStream.range(0, r).parallel().forEach(i -> {
            float[] x = lay.window()[i];
            float[] ls = instanceNorm(x);
            loc[i] = ls[0];
            scale[i] = ls[1];
            for (int p = 0; p < nc; p++) {
                int o = (i * ne + p) * w;
                for (int s = 0; s < ps; s++) {
                    int t = p * ps + s;
                    tok[o + s] = (float) (t - nc * ps) / timeScale;
                    boolean obs = !Float.isNaN(x[t]);
                    tok[o + ps + s] = obs ? scaled(x[t], ls[0], ls[1], c.useArcsinh()) : 0f;
                    tok[o + 2 * ps + s] = obs ? 1f : 0f;
                }
            }
            float[] f = lay.future()[i];
            for (int p = 0; p < h; p++) {
                int o = (i * ne + nc + p) * w;
                for (int s = 0; s < ps; s++) {
                    int t = p * ps + s;
                    tok[o + s] = (float) t / timeScale;
                    boolean obs = !Float.isNaN(f[t]);
                    tok[o + ps + s] = obs ? scaled(f[t], ls[0], ls[1], c.useArcsinh()) : 0f;
                    tok[o + 2 * ps + s] = obs ? 1f : 0f;
                }
            }
        });
        return new ModelPrep(tok, loc, scale);
    }

    /**
     * {@code InstanceNorm} of one row: the mean and the (population) standard deviation of its
     * observed values, a zero deviation floored at {@link #EPS}, and {@code (0, 1)} for a row with
     * none. In double, rounded to float as the reference holds them.
     */
    public static float[] instanceNorm(float[] x) {
        double sum = 0;
        int n = 0;
        for (float v : x) {
            if (!Float.isNaN(v)) {
                sum += v;
                n++;
            }
        }
        if (n == 0) {
            return new float[] {0f, 1f};
        }
        float loc = (float) (sum / n);
        double ss = 0;
        for (float v : x) {
            if (!Float.isNaN(v)) {
                float d = v - loc;
                ss += (double) d * d;
            }
        }
        float scale = (float) Math.sqrt(ss / n);
        return new float[] {loc, scale == 0f ? EPS : scale};
    }

    /** {@code arcsinh((x − loc) / scale)}, in float as the reference computes it. */
    public static float scaled(float x, float loc, float scale, boolean arcsinh) {
        float v = (x - loc) / scale;
        return arcsinh ? asinh(v) : v;
    }

    public static float asinh(float v) {
        double a = Math.abs((double) v);
        double r = Math.log(a + Math.sqrt(a * a + 1.0));
        return (float) (v < 0 ? -r : r);
    }

    /**
     * After the output patch embedding ({@code raw [rows · outputPatches · Q·P]}, each token's
     * quantiles level-major): {@code sinh(x)·scale + loc} per row.
     *
     * @return {@code [rows][Q][outputPatches · P]}
     */
    public float[][][] finish(Chronos2Config c, ModelLayout lay, float[] raw) {
        int r = lay.rows(), ps = c.patchSize(), q = c.numQuantiles(), h = lay.outputPatches();
        float[][][] out = new float[r][q][h * ps];
        IntStream.range(0, r).parallel().forEach(i -> {
            for (int p = 0; p < h; p++) {
                int o = (i * h + p) * q * ps;
                for (int k = 0; k < q; k++) {
                    for (int s = 0; s < ps; s++) {
                        float v = raw[o + k * ps + s];
                        if (c.useArcsinh()) {
                            v = (float) Math.sinh(v);
                        }
                        out[i][k][p * ps + s] = v * scale[i] + loc[i];
                    }
                }
            }
        });
        return out;
    }
}
