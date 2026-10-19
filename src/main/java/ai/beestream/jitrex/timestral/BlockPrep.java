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
package ai.beestream.jitrex.timestral;

import java.util.stream.IntStream;

/**
 * The host ends of {@code predict_block}, shared by every engine: the scaler and the patch tokens
 * in front of the transformer, and the monotone quantiles and denormalization after the head.
 * The scaler needs float64 statistics, which Apple GPUs do not have, and the denormalization
 * clamps infinities, which fast-math device compilers may not preserve; both are
 * {@code O(rows · length)}, and run on the host with the rows in parallel (each row's arithmetic is
 * unchanged, so the results do not depend on the thread count).
 *
 * @param tokens {@code [rows · encoderTokens · 2P]}: each encoder patch's scaled values, then its
 *     missing-value flags, the patch embedding's input
 * @param scaled {@code [rows · window]} the scaled window
 * @param loc {@code [rows · window]} the location each step was scaled with
 * @param scale {@code [rows · window]} the scale each step was scaled with
 */
public record BlockPrep(float[] tokens, float[] scaled, float[] loc, float[] scale) {

    /**
     * The scaler ({@code TimestralModel.scale}, then the whole-row statistics of known covariates)
     * and the embedding input of every encoder token.
     */
    public static BlockPrep of(TimestralConfig c, BlockLayout lay) {
        int r = lay.rows(), ps = c.patchSize(), enc = lay.encoderTokens(), w = enc * ps;
        float[][] window = lay.window();
        boolean[][] obs = lay.observed();
        float[] scaled = new float[r * w], loc = new float[r * w], scale = new float[r * w];
        float[] tok = new float[r * enc * 2 * ps];
        IntStream.range(0, r).parallel().forEach(i -> {
            float[] l = new float[w], s = new float[w];
            Scaler.causalPatchLocScale(window[i], obs[i], ps, c.minimumScale(), l, s);
            boolean known = lay.known(i);
            float[] inst = known ? Scaler.instanceNorm(window[i], obs[i], c.minimumScale()) : null;
            for (int t = 0; t < w; t++) {
                int o = i * w + t;
                if (known) {
                    loc[o] = inst[0];
                    scale[o] = inst[1];
                    scaled[o] = Scaler.affine(window[i][t], inst[0], inst[1], obs[i][t], c.useAsinh() && inst[2] == 0f);
                } else {
                    loc[o] = l[t];
                    scale[o] = s[t];
                    scaled[o] = Scaler.affine(window[i][t], l[t], s[t], obs[i][t], c.useAsinh());
                }
            }
            for (int p = 0; p < enc; p++) {
                int o = (i * enc + p) * 2 * ps;
                for (int st = 0; st < ps; st++) {
                    int t = p * ps + st;
                    tok[o + st] = scaled[i * w + t];
                    tok[o + ps + st] = obs[i][t] ? 0f : 1f;
                }
            }
        });
        return new BlockPrep(tok, scaled, loc, scale);
    }

    /**
     * After the head: makes each step's raw outputs monotone ({@code raw [rows · H·P · Q]}, in place),
     * then denormalizes them with the frame of the last context patch and clamps infinities.
     *
     * @return {@code [rows][H·P][Q]} raw-space quantiles
     */
    public float[][][] finish(TimestralConfig c, BlockLayout lay, Quantiles quantiles, float[] raw) {
        int r = lay.rows(), ps = c.patchSize(), q = c.numQuantiles(), h = lay.horizonPatches();
        int w = lay.encoderTokens() * ps;
        int ll = lay.contextPatches();
        float[][][] out = new float[r][][];
        IntStream.range(0, r).parallel().forEach(i -> {
            for (int v = i * h * ps; v < (i + 1) * h * ps; v++) {
                quantiles.apply(raw, v * q);
            }
            out[i] = new float[h * ps][q];
            int anchor = i * w + (ll - 1) * ps;
            for (int t = 0; t < h * ps; t++) {
                float[] v = out[i][t];
                for (int k = 0; k < q; k++) {
                    v[k] = Scaler.invert(raw[(i * h * ps + t) * q + k], loc[anchor], scale[anchor], c.useAsinh());
                }
                Scaler.clampNonFinite(v);
            }
        });
        return out;
    }
}
