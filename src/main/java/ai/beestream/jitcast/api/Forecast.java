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
package ai.beestream.jitcast.api;

/**
 * Quantile forecasts for one {@link TimeSeries}.
 *
 * @param levels the quantile levels, ascending: 0.1 … 0.9 for TiRex-2, 0.01 … 0.99 for Timestral,
 *     the checkpoint's for Chronos-2
 * @param values {@code values[target][quantile][step]}: one entry per target of the input series,
 *     per level, per forecast step
 */
public record Forecast(float[] levels, float[][][] values) {

    /** {@return the number of forecast targets, as in the input series} */
    public int targets() {
        return values.length;
    }

    /** {@return the number of forecast steps} */
    public int horizon() {
        return values[0][0].length;
    }

    /**
     * {@return the median path of one target, the 0.5 level}
     *
     * @param t the target index
     * @throws IllegalStateException if the model has no 0.5 level
     */
    public float[] median(int t) {
        for (int q = 0; q < levels.length; q++) {
            if (Math.abs(levels[q] - 0.5f) < 1e-6f) {
                return values[t][q];
            }
        }
        throw new IllegalStateException("no 0.5 quantile level");
    }
}
