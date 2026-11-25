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
package ai.beestream.jitcast.toto.tornado;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.toto.TotoConfig;
import ai.beestream.jitcast.toto.TotoGolden;
import ai.beestream.jitcast.toto.TotoPipeline;
import ai.beestream.jitcast.toto.TotoWeights;
import org.junit.Test;

/**
 * The TornadoVM forward pass against the reference: the output head on the reference's exact inputs, and
 * the whole pipeline's forecasts from the raw series, for every golden case. FP32 GEMM modes must match
 * within {@code 1e-4}; modes with FP16 or TF32 GEMM inputs within {@link #REDUCED}.
 */
public class TotoForwardAccelTest {

    private static final double TOLERANCE = 1e-4;
    private static final double REDUCED = 5e-2;

    @Test
    public void headsAndForecastsMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TotoWeights weights = TotoWeights.load(TotoGolden.checkpoint());
                TornadoTotoForward fwd = new TornadoTotoForward(weights.config(), TotoPacked.pack(weights), 4)) {
            TotoConfig c = weights.config();
            double limit = fwd.gemmMode().reducedPrecision() ? REDUCED : TOLERANCE;
            System.out.println("GEMM mode " + fwd.gemmMode() + ", limit " + limit);
            for (String testCase : TotoGolden.cases()) {
                Map<String, Object> info = TotoGolden.caseInfo(testCase);
                int h = TotoGolden.intOf(info, "H");
                TotoPipeline pipeline = new TotoPipeline(c, fwd,
                        TotoPipeline.Options.defaults().withMaxContext(TotoGolden.intOf(info, "context_length")), 3);
                try (SafeTensors g = TotoGolden.open(testCase)) {
                    TotoPipeline.Prepared prep = pipeline.prepare(TotoGolden.assembled(g), h);
                    long t0 = System.nanoTime();
                    float[] head = fwd.forward(prep.input());
                    long ms = (System.nanoTime() - t0) / 1_000_000;
                    double eh = TotoGolden.relativeMaxError(headLayout(head, prep.input().rows(), prep.input().forecast(), c),
                            g.get("head").toFloatArray());
                    double ea = TotoGolden.forecastError(pipeline.finish(prep, head, h), g.get("quantiles"));
                    long t1 = System.nanoTime();
                    List<Forecast> fc = pipeline.forecast(TotoGolden.series(g), h);
                    long ms2 = (System.nanoTime() - t1) / 1_000_000;
                    double fe = TotoGolden.forecastError(fc, g.get("quantiles"));
                    if (!(eh <= limit && ea <= limit && fe <= limit)) {
                        failures.add(testCase + ": head " + eh + " assembled " + ea + " forecast " + fe);
                    }
                    System.out.printf("%s rows=%d patches=%d head %.2e assembled %.2e forecast %.2e  %d / %d ms%n", testCase,
                            prep.input().rows(), prep.input().patches(), eh, ea, fe, ms, ms2);
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    private static float[] headLayout(float[] head, int rows, int k, TotoConfig c) {
        int p = c.patchSize(), nq = c.quantiles();
        float[] out = new float[head.length];
        for (int r = 0; r < rows; r++) {
            for (int f = 0; f < k; f++) {
                for (int j = 0; j < p; j++) {
                    for (int l = 0; l < nq; l++) {
                        out[((l * rows + r) * k + f) * p + j] = head[((r * k + f) * p + j) * nq + l];
                    }
                }
            }
        }
        return out;
    }
}
