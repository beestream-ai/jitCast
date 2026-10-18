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
package ai.beestream.jitrex.timestral.cpu;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitrex.io.SafeTensors;
import ai.beestream.jitrex.timestral.BlockInput;
import ai.beestream.jitrex.timestral.BlockLayout;
import ai.beestream.jitrex.timestral.TimestralConfig;
import ai.beestream.jitrex.timestral.TimestralGolden;
import ai.beestream.jitrex.timestral.TimestralWeights;
import org.junit.Test;

/**
 * The CPU {@code predict_block} against the reference, stage by stage (scaler, patch embedding,
 * every transformer layer, the final norm and the quantile head) and on its output, for every
 * recorded call of every golden case.
 *
 * <p>A stage passes within {@code 1e-4} of the reference, or within {@link #NOISE_FACTOR} times the
 * distance between the reference's own float32 and float64 runs of that call, when that is larger:
 * deep in a long context a single token can be sensitive enough that two float32 implementations
 * scatter around the exact value by more than {@code 1e-4} while the forecasts still agree.
 *
 * <p>With covariates the reference also runs a decoder token for every context patch; nothing reads
 * those tokens, the engine leaves them out, and only the horizon's decoder tokens are compared.
 */
public class BlockGoldenTest {

    private static final double TOLERANCE = 1e-4;
    private static final double NOISE_FACTOR = 4;

    @Test
    public void everyStageMatchesTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TimestralWeights weights = TimestralWeights.load(TimestralGolden.checkpoint())) {
            CpuTimestral ref = new CpuTimestral(weights);
            TimestralConfig c = ref.config();
            int d = c.dModel(), ps = c.patchSize(), q = c.numQuantiles();
            for (String testCase : TimestralGolden.cases()) {
                try (SafeTensors g = TimestralGolden.open(testCase)) {
                    for (int k = 0; k < TimestralGolden.calls(g); k++) {
                        BlockInput block = TimestralGolden.block(g, k, ps);
                        BlockLayout lay = BlockLayout.of(c, block);
                        int r = lay.rows(), enc = lay.encoderTokens(), h = lay.horizonPatches(), ll = lay.contextPatches();
                        // the reference's decoder tokens: h of them, or one per window patch with covariates
                        int refDec = lay.covariates() ? enc : h;
                        int decFrom = lay.covariates() ? ll : 0;
                        Map<String, float[]> stages = new LinkedHashMap<>();
                        long t0 = System.nanoTime();
                        float[][][] out = ref.predictBlock(block, stages::put);
                        long ms = (System.nanoTime() - t0) / 1_000_000;

                        String prefix = "call" + k + ".";
                        double worst = 0;
                        String worstStage = "";
                        for (Map.Entry<String, float[]> e : stages.entrySet()) {
                            String name = e.getKey();
                            float[] expected = g.get(prefix + name).toFloatArray();
                            float[] actual = e.getValue();
                            if (name.startsWith("layer") || name.equals("out_norm")) {
                                int n = enc + h, refN = enc + refDec;
                                expected = concat(TimestralGolden.tokens(expected, r, refN, d, 0, enc),
                                        TimestralGolden.tokens(expected, r, refN, d, enc + decFrom, enc + decFrom + h));
                                actual = concat(TimestralGolden.tokens(actual, r, n, d, 0, enc),
                                        TimestralGolden.tokens(actual, r, n, d, enc, n));
                            } else if (name.equals("head")) {
                                expected = TimestralGolden.tokens(expected, r, refDec * ps, q, decFrom * ps, (decFrom + h) * ps);
                            }
                            double err = TimestralGolden.relativeMaxError(actual, expected);
                            if (err > worst) {
                                worst = err;
                                worstStage = name;
                            }
                            double limit = tolerance(g, prefix, name);
                            if (!(err <= limit)) {
                                failures.add(testCase + " call" + k + " " + name + ": " + err + " > " + limit);
                            }
                        }
                        float[] flat = new float[r * h * ps * q];
                        int i = 0;
                        for (float[][] row : out) {
                            for (float[] v : row) {
                                System.arraycopy(v, 0, flat, i, q);
                                i += q;
                            }
                        }
                        double outErr = TimestralGolden.relativeMaxError(flat, g.get(prefix + "out").toFloatArray());
                        if (!(outErr <= tolerance(g, prefix, "out"))) {
                            failures.add(testCase + " call" + k + " out: " + outErr);
                        }
                        System.out.printf("%-24s call%d rows=%3d tokens=%4d %6d ms  out %.2e  worst stage %s %.2e%n", testCase,
                                k, r, enc + h, ms, outErr, worstStage, worst);
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

    private static float[] concat(float[] a, float[] b) {
        float[] r = new float[a.length + b.length];
        System.arraycopy(a, 0, r, 0, a.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }
}
