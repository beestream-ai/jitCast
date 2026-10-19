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
package ai.beestream.jitcast.cpu;

import java.util.stream.IntStream;

/**
 * Plain-Java FP32 operators shared by the CPU references of every model family, and the oracles of
 * the shared device kernels ({@code tornado.Kernels}). Activations are row-major {@code float[]} of
 * {@code rows × cols}; weights are PyTorch {@code [out, in]}. Reductions accumulate in double so the
 * reference is a stable oracle for the device kernels, not a speed target.
 */
public final class Ops {

    /** torch.finfo(float32).eps: what F.rms_norm uses when no eps is given (the attention qk-norm). */
    public static final float FLOAT32_EPS = Math.ulp(1.0f);
    /** Rows per parallel task in {@link #linear}; small problems stay on the calling thread. */
    private static final int PARALLEL_ROWS = 16;
    /** {@code y[r] = W·x[r] + b}; {@code w} is {@code [out, in]}, {@code b} may be null. */
    public static float[] linear(float[] x, int rows, float[] w, float[] b, int in, int out) {
        float[] y = new float[rows * out];
        IntStream range = IntStream.range(0, (rows + PARALLEL_ROWS - 1) / PARALLEL_ROWS);
        (rows > PARALLEL_ROWS ? range.parallel() : range).forEach(chunk -> {
            int end = Math.min(rows, (chunk + 1) * PARALLEL_ROWS);
            for (int r = chunk * PARALLEL_ROWS; r < end; r++) {
                int xo = r * in;
                for (int o = 0; o < out; o++) {
                    int wo = o * in;
                    double acc = 0;
                    for (int i = 0; i < in; i++) {
                        acc += (double) x[xo + i] * w[wo + i];
                    }
                    y[r * out + o] = (float) (b == null ? acc : acc + b[o]);
                }
            }
        });
        return y;
    }
    /** {@code x · rsqrt(mean(x²) + eps) · w} over each row of {@code cols}; {@code w} may be null. */
    public static float[] rmsNorm(float[] x, int rows, int cols, float[] w, float eps) {
        float[] y = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            int o = r * cols;
            double ss = 0;
            for (int c = 0; c < cols; c++) {
                ss += (double) x[o + c] * x[o + c];
            }
            float inv = (float) (1.0 / Math.sqrt(ss / cols + eps));
            for (int c = 0; c < cols; c++) {
                y[o + c] = w == null ? x[o + c] * inv : x[o + c] * inv * w[c];
            }
        }
        return y;
    }
    public static float silu(float v) {
        return (float) (v / (1.0 + Math.exp(-v)));
    }
    public static double sigmoid(double v) {
        return 1.0 / (1.0 + Math.exp(-v));
    }
    public static void siluInPlace(float[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = silu(x[i]);
        }
    }
    public static void reluInPlace(float[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.max(x[i], 0f);
        }
    }
    public static void addInPlace(float[] acc, float[] x) {
        for (int i = 0; i < acc.length; i++) {
            acc[i] += x[i];
        }
    }

    private Ops() {}
}
