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

/**
 * What {@code Chronos2Model.forward} receives at inference: one row per target and covariate of
 * every series, left-padded to a common length, and the number of patches to forecast.
 *
 * @param context {@code [R][T]} context values, {@code NaN} where missing (and in the padding)
 * @param group {@code [R]} group id of each row; the rows of one group attend each other
 * @param futureCovariates {@code [R][≤ outputPatches · P]} the known future values of each row,
 *     {@code NaN} where unknown (every target and past-only covariate), or null
 * @param outputPatches the patches to forecast
 */
public record ModelInput(float[][] context, int[] group, float[][] futureCovariates, int outputPatches) {

    public int rows() {
        return context.length;
    }

    public int length() {
        return context[0].length;
    }
}
