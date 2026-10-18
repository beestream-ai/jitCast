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
package ai.beestream.jitrex.timestral.tornado;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitrex.api.Forecast;
import ai.beestream.jitrex.io.SafeTensors;
import ai.beestream.jitrex.timestral.BlockInput;
import ai.beestream.jitrex.timestral.PipelineGoldenTest;
import ai.beestream.jitrex.timestral.TimestralConfig;
import ai.beestream.jitrex.timestral.TimestralGolden;
import ai.beestream.jitrex.timestral.TimestralPipeline;
import ai.beestream.jitrex.timestral.TimestralWeights;
import org.junit.Test;

/**
 * The TornadoVM {@code predict_block} on every recorded call, and the TornadoVM engine end to end on
 * every golden case, against the reference. Pick the device with
 * {@code -Dtirex.tornado.device=<driver>:<device>}.
 */
public class TimestralForwardAccelTest {

    private static final double TOLERANCE = 1e-4;
    /**
     * With FP16 or TF32 GEMM inputs ({@code -Dtimestral.tornado.gemm=half|cublas-fp16|cublas-tf32}): a
     * bound for a smoke check, not parity. Measured on Timestral-5M with FP16 GEMMs, the long
     * multi-block horizons are within 4e-2 of the reference's range and the rest within 1e-2.
     */
    private static final double REDUCED_TOLERANCE = 5e-2;

    @Test
    public void everyBlockMatchesTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TimestralWeights weights = TimestralWeights.load(TimestralGolden.checkpoint());
                TimestralForward fwd = new TimestralForward(weights.config(), TimestralPacked.pack(weights))) {
            TimestralConfig c = weights.config();
            int ps = c.patchSize(), q = c.numQuantiles();
            double base = fwd.gemmMode().reducedPrecision() ? REDUCED_TOLERANCE : TOLERANCE;
            for (String testCase : TimestralGolden.cases()) {
                try (SafeTensors g = TimestralGolden.open(testCase)) {
                    for (int k = 0; k < TimestralGolden.calls(g); k++) {
                        BlockInput block = TimestralGolden.block(g, k, ps);
                        fwd.predictBlock(block); // compile and warm up
                        long t0 = System.nanoTime();
                        float[][][] out = fwd.predictBlock(block);
                        double ms = (System.nanoTime() - t0) / 1e6;
                        float[] flat = new float[out.length * out[0].length * q];
                        int i = 0;
                        for (float[][] row : out) {
                            for (float[] v : row) {
                                System.arraycopy(v, 0, flat, i, q);
                                i += q;
                            }
                        }
                        String key = "call" + k + ".fp32err.out";
                        double limit = Math.max(base, 4 * (g.contains(key) ? g.get(key).getFloat(0) : 0));
                        double err = TimestralGolden.relativeMaxError(flat, g.get("call" + k + ".out").toFloatArray());
                        System.out.printf("%-24s call%d rows=%3d %8.2f ms  out rel max err %.2e%n", testCase, k, block.rows(), ms, err);
                        if (!(err <= limit)) {
                            failures.add(testCase + " call" + k + ": " + err);
                        }
                    }
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    @Test
    public void forecastsMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TornadoTimestralEngine engine = TornadoTimestralEngine.load(TimestralGolden.checkpoint())) {
            TimestralConfig c = engine.config();
            double limit = engine.forwardPass().gemmMode().reducedPrecision() ? REDUCED_TOLERANCE : TOLERANCE;
            for (String testCase : TimestralGolden.cases()) {
                Map<String, Object> info = TimestralGolden.caseInfo(testCase);
                int h = ((Number) info.get("H")).intValue();
                try (SafeTensors g = TimestralGolden.open(testCase)) {
                    TimestralPipeline p = engine.pipeline().withOptions(PipelineGoldenTest.options(c, info));
                    List<Forecast> fc = p.forecast(PipelineGoldenTest.series(g, false), h);
                    double err = PipelineGoldenTest.forecastError(fc, g.get("forecast"));
                    System.out.printf("%-24s H=%5d forecast rel max err %.2e%n", testCase, h, err);
                    if (!(err <= limit)) {
                        failures.add(testCase + " forecast: " + err);
                    }
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }
}
