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
package ai.beestream.jitcast.timesfm.cpu;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.timesfm.PatchForward;
import ai.beestream.jitcast.timesfm.TimesFmConfig;
import ai.beestream.jitcast.timesfm.TimesFmGolden;
import ai.beestream.jitcast.timesfm.TimesFmWeights;
import org.junit.Test;

/**
 * The CPU forward pass against the reference's prefill, stage by stage (the tokenizer, every layer, the
 * point head of every patch and both heads of the last patch), for every recorded decode call of every
 * golden case. Activations are compared on the observed patches.
 *
 * <p>A stage passes within {@code 1e-4} of the reference, or within {@link #NOISE_FACTOR} times the
 * distance between the reference's own float32 and float64 runs of that call, when that is larger.
 */
public class ForwardGoldenTest {

    private static final double TOLERANCE = 1e-4;
    private static final double NOISE_FACTOR = 4;

    @Test
    public void everyStageMatchesTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TimesFmWeights weights = TimesFmWeights.load(TimesFmGolden.checkpoint())) {
            CpuTimesFm ref = new CpuTimesFm(weights);
            TimesFmConfig c = weights.config();
            int p = c.patchLength(), d = c.hiddenSize(), q = c.outputs();
            for (String testCase : TimesFmGolden.cases()) {
                try (SafeTensors g = TimesFmGolden.open(testCase)) {
                    for (int k = 0; k < TimesFmGolden.calls(g); k++) {
                        PatchForward.Batch in = TimesFmGolden.prefill(g, k, p);
                        boolean[][] obs = TimesFmGolden.observed(in, p);
                        long t0 = System.nanoTime();
                        CpuTimesFm.Stages s = ref.stages(in);
                        long ms = (System.nanoTime() - t0) / 1_000_000;
                        String prefix = "call" + k + ".";
                        double worst = 0;
                        String worstStage = "";
                        List<String> names = new ArrayList<>(List.of("tokens"));
                        List<float[]> values = new ArrayList<>(List.of(s.tokens()));
                        for (int i = 0; i < s.layers().size(); i++) {
                            names.add("layer" + i);
                            values.add(s.layers().get(i));
                        }
                        names.add("head_point");
                        values.add(s.point());
                        for (int i = 0; i < names.size(); i++) {
                            String name = names.get(i);
                            int width = name.equals("head_point") ? c.horizonLength() * q : d;
                            float[] expected = TimesFmGolden.observedPatches(g.get(prefix + name).toFloatArray(), obs, width);
                            float[] actual = TimesFmGolden.observedPatches(values.get(i), obs, width);
                            double err = TimesFmGolden.relativeMaxError(actual, expected);
                            if (err > worst) {
                                worst = err;
                                worstStage = name;
                            }
                            double limit = Math.max(TOLERANCE, NOISE_FACTOR * TimesFmGolden.noise(g, prefix + "fp32err." + name));
                            if (!(err <= limit)) {
                                failures.add(testCase + " call" + k + " " + name + ": " + err + " > " + limit);
                            }
                        }
                        double spreadErr = TimesFmGolden.relativeMaxError(s.heads().spread(),
                                g.get(prefix + "head_spread").toFloatArray());
                        double spreadLimit = Math.max(TOLERANCE, NOISE_FACTOR * TimesFmGolden.noise(g, prefix + "fp32err.head_spread"));
                        if (!(spreadErr <= spreadLimit)) {
                            failures.add(testCase + " call" + k + " head_spread: " + spreadErr + " > " + spreadLimit);
                        }
                        System.out.printf("%s call%d rows=%d patches=%d worst %.2e (%s) spread %.2e  %d ms%n", testCase, k,
                                in.rows(), in.patches(), worst, worstStage, spreadErr, ms);
                    }
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }
}
