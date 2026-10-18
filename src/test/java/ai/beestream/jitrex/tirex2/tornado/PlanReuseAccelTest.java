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
package ai.beestream.jitrex.tirex2.tornado;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitrex.io.SafeTensors;
import ai.beestream.jitrex.tirex2.GoldenData;
import ai.beestream.jitrex.tirex2.ModelBatch;
import ai.beestream.jitrex.tirex2.TiRex2Weights;
import ai.beestream.jitrex.tirex2.cpu.CpuReference;
import org.junit.Test;

/**
 * A cached plan must give the right answer for every call, not only the first: runs one plan on
 * several different 1-row batches (the same shape) and checks each against the CPU reference.
 */
public class PlanReuseAccelTest {

    @Test
    public void everyCallOfACachedPlanIsCorrect() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TiRex2Weights w = TiRex2Weights.load(GoldenData.weights())) {
            CpuReference cpu = new CpuReference(w);
            try (TornadoForward tf = new TornadoForward(w.config(), PackedWeights.pack(w))) {
                for (String c : List.of("ref_univariate", "short_univariate", "full_context", "binary", "ref_univariate")) {
                    try (SafeTensors g = GoldenData.open(c)) {
                        ModelBatch b = GoldenData.batch(g, 0);
                        double err = GoldenData.relativeMaxError(TornadoForwardAccelTest.flatten(tf.forward(b)),
                                TornadoForwardAccelTest.flatten(cpu.forward(b)));
                        System.out.printf("%-18s plans=%d rel %.2e%n", c, tf.cachedShapes().size(), err);
                        // the limits of TornadoForwardAccelTest: 1e-4 in FP32, NXAI's GPU error otherwise
                        if (err > (TornadoForward.reducedPrecision() ? GoldenData.nxaiGpuError(c)[0] : 1e-4)) {
                            failures.add(c + ": " + err);
                        }
                    }
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }
}
