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
package ai.beestream.jitcast.chronos2.tornado;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.chronos2.Chronos2Config;
import ai.beestream.jitcast.chronos2.Chronos2Golden;
import ai.beestream.jitcast.chronos2.Chronos2Pipeline;
import ai.beestream.jitcast.chronos2.Chronos2Weights;
import ai.beestream.jitcast.chronos2.ModelInput;
import ai.beestream.jitcast.io.SafeTensors;
import org.junit.Test;

/**
 * The TornadoVM forward pass on every recorded call, and the TornadoVM engine end to end on every
 * golden case, against the reference. Pick the device with
 * {@code -Djitcast.tornado.device=<driver>:<device>}.
 */
public class Chronos2ForwardAccelTest {

    private static final double TOLERANCE = 1e-4;
    /** With FP16 or TF32 GEMM inputs ({@code -Dchronos2.tornado.gemm=half|cublas-fp16|cublas-tf32}): a smoke check, not parity. */
    private static final double REDUCED_TOLERANCE = 5e-2;

    @Test
    public void everyCallMatchesTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (Chronos2Weights weights = Chronos2Weights.load(Chronos2Golden.checkpoint());
                Chronos2Forward fwd = new Chronos2Forward(weights.config(), Chronos2Packed.pack(weights))) {
            Chronos2Config c = weights.config();
            double base = fwd.gemmMode().reducedPrecision() ? REDUCED_TOLERANCE : TOLERANCE;
            for (String testCase : Chronos2Golden.cases()) {
                try (SafeTensors g = Chronos2Golden.open(testCase)) {
                    for (int k = 0; k < Chronos2Golden.calls(g); k++) {
                        ModelInput in = Chronos2Golden.input(g, k, c.patchSize());
                        fwd.forward(in); // compile and warm up
                        long t0 = System.nanoTime();
                        float[][][] out = fwd.forward(in);
                        double ms = (System.nanoTime() - t0) / 1e6;
                        String key = "call" + k + ".fp32err.out";
                        double limit = Math.max(base, 4 * (g.contains(key) ? g.get(key).getFloat(0) : 0));
                        double err = Chronos2Golden.relativeMaxError(Chronos2Golden.flat(out),
                                g.get("call" + k + ".out").toFloatArray());
                        System.out.printf("%-22s call%d rows=%4d %8.2f ms  out rel max err %.2e%n", testCase, k, in.rows(), ms, err);
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
        try (TornadoChronos2Engine engine = TornadoChronos2Engine.load(Chronos2Golden.checkpoint())) {
            Chronos2Config c = engine.config();
            double limit = engine.forwardPass().gemmMode().reducedPrecision() ? REDUCED_TOLERANCE : TOLERANCE;
            for (String testCase : Chronos2Golden.cases()) {
                Map<String, Object> info = Chronos2Golden.caseInfo(testCase);
                int h = Chronos2Golden.horizon(info);
                try (SafeTensors g = Chronos2Golden.open(testCase)) {
                    Chronos2Pipeline p = engine.pipeline().withOptions(Chronos2Golden.options(c, info));
                    List<Forecast> fc = p.forecast(Chronos2Golden.series(g, info), h);
                    double err = Chronos2Golden.forecastError(fc, g.get("forecast"));
                    System.out.printf("%-22s H=%5d forecast rel max err %.2e%n", testCase, h, err);
                    if (!(err <= limit)) {
                        failures.add(testCase + " forecast: " + err);
                    }
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }
}
