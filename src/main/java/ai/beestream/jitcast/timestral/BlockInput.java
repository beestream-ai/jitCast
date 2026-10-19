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
package ai.beestream.jitcast.timestral;

/**
 * What {@code TimestralModel.predict_block} receives: a packed context of one row per target and
 * covariate of every series, and the number of patches to forecast after it.
 *
 * @param inputs {@code [R][T]} context values, {@code T} a multiple of the patch size
 * @param observed {@code [R][T]} the observation mask
 * @param group {@code [R]} series id of each row; rows of one series attend each other
 * @param horizonPatches the patches to forecast
 * @param role {@code [R]} {@link #TARGET}, {@link #PAST} or {@link #KNOWN} per row, or null when every
 *     row is a target
 * @param knownFuture {@code [R][horizonPatches · P]} known-future covariate values over the block,
 *     zeros on other rows, or null
 * @param knownFutureMask the observation mask of {@code knownFuture}, or null
 */
public record BlockInput(float[][] inputs, boolean[][] observed, int[] group, int horizonPatches, int[] role,
        float[][] knownFuture, boolean[][] knownFutureMask) {

    public static final int TARGET = 0;
    public static final int PAST = 1;
    public static final int KNOWN = 2;

    public int rows() {
        return inputs.length;
    }

    public int length() {
        return inputs[0].length;
    }
}
