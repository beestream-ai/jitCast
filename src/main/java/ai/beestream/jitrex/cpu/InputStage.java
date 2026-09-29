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
package ai.beestream.jitrex.cpu;

import ai.beestream.jitrex.model.TiRex2Config;

/**
 * The model's input stage on the batch {@code TiRex2.forward()} receives: scale each row
 * ({@code Scaler.scale}), cut it into patches ({@code Tokenizer.input_transform}) and append the
 * is-present mask, giving the {@code [rows, patches, 2·patch]} input of the patch embedding.
 */
public final class InputStage {

    /** Per-row scaler state, needed again to undo the scaling on the output. */
    public record ScalerState(float[] loc, float[] scale, boolean[] binary) {}

    /** The embedding input {@code tokens [rows · patches · 2·patch]} and the scaler state. */
    public record Result(float[] tokens, int rows, int patches, ScalerState scaler) {}

    private InputStage() {}

    /** @param x {@code [rows][paddedLength]}, NaN where there is no observation */
    public static Result apply(float[][] x, TiRex2Config c) {
        int rows = x.length;
        int len = c.paddedLength();
        int p = c.patchSize();
        int patches = c.numPatches();
        float[] loc = new float[rows];
        float[] scale = new float[rows];
        boolean[] binary = new boolean[rows];
        float[] tokens = new float[rows * patches * 2 * p];

        for (int r = 0; r < rows; r++) {
            float[] row = x[r];
            if (row.length != len) {
                throw new IllegalArgumentException("row " + r + " has length " + row.length + ", expected " + len);
            }
            // loc = nan_to_num(nanmean(x), 0); scale = clamp(nan_to_num(sqrt(nanmean((x-loc)²)), 1), eps)
            double sum = 0;
            int n = 0;
            boolean isBinary = true;
            for (float v : row) {
                if (!Float.isNaN(v)) {
                    sum += v;
                    n++;
                    isBinary &= v == 0f || v == 1f;
                }
            }
            float l = n == 0 ? 0f : (float) (sum / n);
            double ss = 0;
            for (float v : row) {
                if (!Float.isNaN(v)) {
                    double dv = v - l;
                    ss += dv * dv;
                }
            }
            float s = n == 0 ? 1f : (float) Math.sqrt(ss / n);
            s = Math.max(s, TiRex2Config.SCALER_EPS);
            if (c.binaryAware() && isBinary) {
                l = 0f;
                s = 1f;
            }
            loc[r] = l;
            scale[r] = s;
            binary[r] = c.binaryAware() && isBinary;
            boolean arcsinh = c.useArcsinh() && !binary[r];

            for (int t = 0; t < patches; t++) {
                int base = (r * patches + t) * 2 * p;
                for (int i = 0; i < p; i++) {
                    float v = row[t * p + i];
                    boolean present = !Float.isNaN(v);
                    float z = (v - l) / s;
                    if (arcsinh) {
                        z = (float) asinh(z);
                    }
                    tokens[base + i] = present ? z : 0f;
                    tokens[base + p + i] = present ? 1f : 0f;
                }
            }
        }
        return new Result(tokens, rows, patches, new ScalerState(loc, scale, binary));
    }

    static double asinh(double z) {
        // log(z + sqrt(z²+1)), written to stay accurate for negative z
        double a = Math.abs(z);
        double r = Math.log(a + Math.sqrt(a * a + 1.0));
        return z < 0 ? -r : r;
    }
}
