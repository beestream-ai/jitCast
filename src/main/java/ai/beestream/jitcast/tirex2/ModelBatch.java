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
package ai.beestream.jitcast.tirex2;

/**
 * The batch {@code TiRex2.forward()} receives: every target and covariate row of every series,
 * padded to the model length.
 *
 * @param x {@code [rows][contextLen + futureLen]}, NaN where there is no observation
 * @param group group id of each row; rows attend across variates only within their group
 * @param target whether each row is a target (covariate rows cannot attend to targets)
 */
public record ModelBatch(float[][] x, float[] group, boolean[] target) {

    public int rows() {
        return x.length;
    }
}
