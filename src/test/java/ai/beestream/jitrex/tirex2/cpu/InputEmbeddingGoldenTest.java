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
package ai.beestream.jitrex.tirex2.cpu;

import static org.junit.Assert.assertTrue;

import java.io.IOException;

import ai.beestream.jitrex.io.SafeTensors;
import ai.beestream.jitrex.tirex2.GoldenData;
import ai.beestream.jitrex.tirex2.ModelBatch;
import ai.beestream.jitrex.tirex2.TiRex2Weights;
import org.junit.Test;

/** Scaler + patching + patch embedding against {@code input_patch_embedding} in every golden case. */
public class InputEmbeddingGoldenTest {

    private static final double TOLERANCE = 1e-4;

    @Test
    public void matchesReferenceInEveryCase() throws IOException {
        try (TiRex2Weights weights = TiRex2Weights.load(GoldenData.weights())) {
            CpuReference ref = new CpuReference(weights);
            for (String testCase : GoldenData.cases()) {
                try (SafeTensors g = GoldenData.open(testCase)) {
                    ModelBatch batch = GoldenData.batch(g, 0);
                    int rows = batch.x().length;
                    float[] actual = ref.embed(InputStage.apply(batch.x(), ref.config()));
                    float[] expected = g.get("act.pass0.input_patch_embedding#0.out").toFloatArray();
                    double err = GoldenData.relativeMaxError(actual, expected);
                    System.out.printf("%-26s rows=%3d  rel max err %.2e%n", testCase, rows, err);
                    assertTrue(testCase + ": relative error " + err, err < TOLERANCE);
                }
            }
        }
    }
}
