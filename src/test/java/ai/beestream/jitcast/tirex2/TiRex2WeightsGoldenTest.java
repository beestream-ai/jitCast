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

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.List;

import ai.beestream.jitcast.tirex2.TiRex2Config.BlockType;
import org.junit.Test;

public class TiRex2WeightsGoldenTest {

    @Test
    public void bindsEveryTensorOfTheReleasedCheckpoint() throws IOException {
        try (TiRex2Weights w = TiRex2Weights.load(GoldenData.weights())) {
            TiRex2Config c = w.config();
            assertEquals(74, c.numPatches());
            assertEquals(512, c.embeddingDim());
            assertEquals(1408, c.ffnDim());
            assertEquals(9, c.numQuantiles());
            assertEquals(12, w.blocks().size());
            for (int i = 0; i < 12; i++) {
                assertEquals(i % 2 == 0 ? BlockType.MLSTM : BlockType.SLSTM, c.recipe().get(i));
            }
            assertEquals(List.of(BlockType.MLSTM, BlockType.SLSTM), c.recipe().subList(0, 2));
            // 38.4M univariate + 44.1M multivariate parameters, each tensor counted once
            assertEquals(82_501_744L, w.parameterCount());
        }
    }
}
