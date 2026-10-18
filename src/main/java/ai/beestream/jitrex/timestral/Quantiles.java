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

/**
 * The monotone quantile head ({@code monotone_quantiles}): the median channel is free and the
 * other levels fan out from it as {@code median ∓ cumsum(widthFactor · Δp · softplus(raw))}, with
 * {@code Δp} the probability gap to the next level inward, so the quantiles never cross.
 */
public final class Quantiles {

    private final int median;
    /** {@code widthFactor · Δp} per level, rounded to float as the reference's tensor holds it. */
    private final float[] scale;

    public Quantiles(TimestralConfig config) {
        double[] levels = config.quantileKnots();
        median = config.medianIndex();
        scale = new float[levels.length];
        for (int i = 0; i < levels.length; i++) {
            double width = i > median ? levels[i] - levels[i - 1] : i < median ? levels[i + 1] - levels[i] : 0.0;
            scale[i] = (float) width;
        }
        float wf = config.quantileWidthFactor();
        for (int i = 0; i < scale.length; i++) {
            scale[i] = wf * scale[i];
        }
    }

    /** {@code width_factor · Δp} per level, for the device kernel. */
    public float[] scale() {
        return scale.clone();
    }

    public int median() {
        return median;
    }

    /** Turns the raw head outputs {@code raw[off .. off+Q)} into non-crossing quantiles, in place. */
    public void apply(float[] raw, int off) {
        int q = scale.length;
        float m = raw[off + median];
        float[] inc = new float[q];
        for (int i = 0; i < q; i++) {
            inc[i] = scale[i] * softplus(raw[off + i]);
        }
        float up = 0;
        for (int i = median + 1; i < q; i++) {
            up += inc[i];
            raw[off + i] = clamp(m + up);
        }
        float down = 0;
        for (int i = median - 1; i >= 0; i--) {
            down += inc[i];
            raw[off + i] = clamp(m - down);
        }
        raw[off + median] = clamp(m);
    }

    /** {@code torch.nn.functional.softplus}: {@code log1p(exp(x))}, or {@code x} above 20. */
    public static float softplus(float x) {
        return x > 20f ? x : (float) Math.log1p(Math.exp(x));
    }

    private static float clamp(float v) {
        return Math.max(-Float.MAX_VALUE, Math.min(Float.MAX_VALUE, v));
    }
}
