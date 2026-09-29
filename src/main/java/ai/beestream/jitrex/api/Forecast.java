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
package ai.beestream.jitrex.api;

/**
 * Quantile forecasts for one {@link TimeSeries}: {@code values[target][quantile][step]}, with
 * the quantile levels in {@link #levels()} (0.1 … 0.9 for TiRex-2).
 */
public record Forecast(float[] levels, float[][][] values) {

    public int targets() {
        return values.length;
    }

    public int horizon() {
        return values[0][0].length;
    }

    /** The median path of target {@code t} (the 0.5 level). */
    public float[] median(int t) {
        for (int q = 0; q < levels.length; q++) {
            if (Math.abs(levels[q] - 0.5f) < 1e-6f) {
                return values[t][q];
            }
        }
        throw new IllegalStateException("no 0.5 quantile level");
    }
}
