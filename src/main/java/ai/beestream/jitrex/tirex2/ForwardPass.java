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
package ai.beestream.jitrex.tirex2;

/** One engine's {@code TiRex2.forward()}: the rescaled quantile outputs over the whole padded length. */
@FunctionalInterface
public interface ForwardPass {

    /** @return {@code [rows][quantiles][contextLen + futureLen]} */
    float[][][] forward(ModelBatch batch);
}
