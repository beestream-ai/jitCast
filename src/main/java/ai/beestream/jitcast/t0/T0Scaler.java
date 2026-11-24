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

/**
 * The reference's {@code CausalScaler}: targets and past-only covariates are standardized at every step by
 * the running (Welford) mean and deviation of their observations so far, known-future covariates by their
 * row's overall mean and deviation, then {@code asinh} is applied. Computed in float32 in the reference's
 * order (its running sums accumulate in double, as PyTorch's {@code cumsum} does on the CPU), so the scaled
 * inputs match the reference's to the last bit or two.
 */
public final class T0Scaler {

    /**
     * A scaled window and the statistics it was scaled with.
     *
     * @param scaled {@code [rows · width]} the model's input values
     * @param loc {@code [rows · width]} the location of each cell
     * @param scale {@code [rows · width]} the scale of each cell
     */
    public record Scaled(float[] scaled, float[] loc, float[] scale) {}

    private final T0Config config;

    /**
     * @param config the checkpoint's configuration, which fixes the scaler's settings
     */
    public T0Scaler(T0Config config) {
        this.config = config;
    }

    /**
     * {@return the scaled window}
     *
     * @param b the window
     */
    public Scaled scale(T0Batch b) {
        int w = b.width();
        float eps = config.scalerEps();
        float[] scaled = new float[b.rows() * w], loc = new float[scaled.length], scale = new float[scaled.length];
        float[] cLoc = new float[w], cScale = new float[w];
        for (int r = 0; r < b.rows(); r++) {
            int o = r * w;
            // The causal statistics.
            double count = 0, sum = 0, m2 = 0;
            float prevMean = 0f;
            for (int t = 0; t < w; t++) {
                float x = value(b, o + t);
                boolean invalid = invalid(b, o + t);
                float valid = invalid ? 0f : 1f, masked = invalid ? 0f : x;
                count += valid;
                float n = Math.max((float) count, 1f);
                sum += masked;
                float mean = (float) sum / n;
                float term = (masked - prevMean) * (masked - mean) * valid;
                m2 += term;
                float variance = Math.max((float) m2, 0f) / Math.max(n - 1f, 1f);
                cLoc[t] = mean;
                cScale[t] = config.scalerEpsVarianceOffset() ? (float) Math.sqrt(variance + eps)
                        : Math.max((float) Math.sqrt(variance), eps);
                prevMean = mean;
            }
            // The global statistics.
            float gCount = 0f, gSum = 0f;
            for (int t = 0; t < w; t++) {
                boolean invalid = invalid(b, o + t);
                gCount += invalid ? 0f : 1f;
                gSum += invalid ? 0f : value(b, o + t);
            }
            float gMean = gSum / Math.max(gCount, 1f), gSq = 0f;
            for (int t = 0; t < w; t++) {
                float dx = value(b, o + t) - gMean;
                gSq += invalid(b, o + t) ? 0f : dx * dx;
            }
            float gStd = Math.max((float) Math.sqrt(gSq / Math.max(gCount, 2f)), eps);
            for (int t = 0; t < w; t++) {
                int i = o + t, g = b.group()[i], type = b.type()[i];
                boolean real = g >= 0;
                boolean future = real && type == T0Batch.FUTURE;
                boolean causal = real && (type == T0Batch.TARGET || type == T0Batch.HISTORICAL);
                loc[i] = future ? gMean : causal ? cLoc[t] : 0f;
                scale[i] = future ? gStd : causal ? cScale[t] : 1f;
                float s = (value(b, i) - loc[i]) / scale[i];
                if (config.scalerUseArcsinh()) {
                    s = asinh(s);
                }
                if (config.scalerZeroNoValue()) {
                    byte m = b.mask()[i];
                    if (m == T0Batch.PAD || m == T0Batch.MISSING || m == T0Batch.WITHHELD) {
                        s = 0f;
                    }
                }
                scaled[i] = s;
            }
        }
        return new Scaled(scaled, loc, scale);
    }

    /**
     * {@return a scaled prediction back in data space: {@code sinh(v) · scale + loc}, or {@code v · scale + loc}
     * without {@code asinh}}
     *
     * @param v the model's output
     * @param loc the location of the patch's last step
     * @param scale the scale of the patch's last step
     */
    public float rescale(float v, float loc, float scale) {
        float x = config.scalerUseArcsinh() ? (float) Math.sinh(v) : v;
        return x * scale + loc;
    }

    private float value(T0Batch b, int i) {
        float v = b.values()[i];
        return !config.scalerZeroNoValue() && Float.isNaN(v) ? 0f : v;
    }

    private static boolean invalid(T0Batch b, int i) {
        return b.mask()[i] != T0Batch.VALID || Float.isNaN(b.values()[i]);
    }

    /** {@return {@code asinh(x)} rounded to float, accurate near 0 and for large magnitudes} */
    static float asinh(float x) {
        double a = Math.abs((double) x);
        double y = a > 1e8 ? Math.log(a) + Math.log(2) : Math.log1p(a + a * a / (1 + Math.sqrt(1 + a * a)));
        return (float) Math.copySign(y, x);
    }
}
