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
package ai.beestream.jitcast.chronos2.cpu;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.chronos2.Chronos2Config;
import ai.beestream.jitcast.chronos2.Chronos2Golden;
import ai.beestream.jitcast.chronos2.Chronos2Weights;
import ai.beestream.jitcast.chronos2.ModelInput;
import ai.beestream.jitcast.io.SafeTensors;
import org.junit.Test;

/**
 * The CPU forward pass against the reference, stage by stage (instance norm, patch embeddings,
 * every encoder block, the final norm and the output patch embedding) and on its output, for every
 * recorded call of every golden case.
 *
 * <p>A stage passes within {@code 1e-4} of the reference, or within {@link #NOISE_FACTOR} times the
 * distance between the reference's own float32 and float64 runs of that call, when that is larger.
 * The encoder's activations are compared on the valid tokens only (see {@code ModelLayout}).
 */
public class ForwardGoldenTest {

    private static final double TOLERANCE = 1e-4;
    private static final double NOISE_FACTOR = 4;

    @Test
    public void everyStageMatchesTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (Chronos2Weights weights = Chronos2Weights.load(Chronos2Golden.checkpoint())) {
            CpuChronos2 ref = new CpuChronos2(weights);
            Chronos2Config c = ref.config();
            int d = c.dModel();
            for (String testCase : Chronos2Golden.cases()) {
                try (SafeTensors g = Chronos2Golden.open(testCase)) {
                    for (int k = 0; k < Chronos2Golden.calls(g); k++) {
                        ModelInput in = Chronos2Golden.input(g, k, c.patchSize());
                        boolean[][] live = Chronos2Golden.live(g, k);
                        Map<String, float[]> stages = new LinkedHashMap<>();
                        long t0 = System.nanoTime();
                        float[][][] out = ref.forward(in, stages::put);
                        long ms = (System.nanoTime() - t0) / 1_000_000;

                        String prefix = "call" + k + ".";
                        double worst = 0;
                        String worstStage = "";
                        for (Map.Entry<String, float[]> e : stages.entrySet()) {
                            String name = e.getKey();
                            float[] expected = g.get(prefix + name).toFloatArray();
                            float[] actual = e.getValue();
                            if (name.startsWith("layer") || name.equals("final_norm")) {
                                expected = Chronos2Golden.liveTokens(expected, live, d);
                                actual = Chronos2Golden.liveTokens(actual, live, d);
                            }
                            double err = Chronos2Golden.relativeMaxError(actual, expected);
                            if (err > worst) {
                                worst = err;
                                worstStage = name;
                            }
                            double limit = tolerance(g, prefix, name);
                            if (!(err <= limit)) {
                                failures.add(testCase + " call" + k + " " + name + ": " + err + " > " + limit);
                            }
                        }
                        double outErr = Chronos2Golden.relativeMaxError(Chronos2Golden.flat(out),
                                g.get(prefix + "out").toFloatArray());
                        if (!(outErr <= tolerance(g, prefix, "out"))) {
                            failures.add(testCase + " call" + k + " out: " + outErr);
                        }
                        System.out.printf("%-22s call%d rows=%4d tokens=%4d %6d ms  out %.2e  worst stage %s %.2e%n", testCase,
                                k, in.rows(), live[0].length, ms, outErr, worstStage, worst);
                    }
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    /** {@code max(1e-4, NOISE_FACTOR · the reference's own float32 error on this stage)}. */
    private static double tolerance(SafeTensors g, String prefix, String stage) {
        String key = prefix + "fp32err." + stage;
        double noise = g.contains(key) ? g.get(key).getFloat(0) : 0;
        return Math.max(TOLERANCE, NOISE_FACTOR * noise);
    }
}
