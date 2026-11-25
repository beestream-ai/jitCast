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
package ai.beestream.jitcast.toto.cpu;

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
 * The CPU forward pass against the reference, stage by stage (the patch embedding, every layer, the final
 * norm and the output head), on exactly the inputs the reference's forward saw, and the forecasts those
 * stages lead to.
 *
 * <p>A stage passes within {@code 1e-4} of the reference, or within {@link #NOISE_FACTOR} times the
 * distance between the reference's own float32 and float64 runs, when that is larger.
 */
public class ForwardGoldenTest {

    private static final double TOLERANCE = 1e-4;
    private static final double NOISE_FACTOR = 4;

    @Test
    public void everyStageMatchesTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TotoWeights weights = TotoWeights.load(TotoGolden.checkpoint())) {
            CpuToto cpu = new CpuToto(weights);
            TotoConfig c = weights.config();
            for (String testCase : TotoGolden.cases()) {
                Map<String, Object> info = TotoGolden.caseInfo(testCase);
                int h = TotoGolden.intOf(info, "H");
                TotoPipeline pipeline = new TotoPipeline(c, cpu,
                        TotoPipeline.Options.defaults().withMaxContext(TotoGolden.intOf(info, "context_length")), 1 << 20);
                try (SafeTensors g = TotoGolden.open(testCase)) {
                    TotoPipeline.Prepared prep = pipeline.prepare(TotoGolden.assembled(g), h);
                    long t0 = System.nanoTime();
                    CpuToto.Stages s = cpu.stages(prep.input());
                    long ms = (System.nanoTime() - t0) / 1_000_000;
                    List<String> names = new ArrayList<>(List.of("tokens"));
                    List<float[]> values = new ArrayList<>(List.of(s.tokens()));
                    for (int i = 0; i < s.layers().size(); i++) {
                        names.add("layer" + i);
                        values.add(s.layers().get(i));
                    }
                    names.add("out_norm");
                    values.add(s.outNorm());
                    names.add("head");
                    values.add(headLayout(s.head(), prep.input().rows(), prep.input().forecast(), c));
                    double worst = 0;
                    String worstStage = "";
                    for (int i = 0; i < names.size(); i++) {
                        String name = names.get(i);
                        double err = TotoGolden.relativeMaxError(values.get(i), g.get(name).toFloatArray());
                        if (err > worst) {
                            worst = err;
                            worstStage = name;
                        }
                        double limit = Math.max(TOLERANCE, NOISE_FACTOR * TotoGolden.noise(g, name));
                        if (!(err <= limit)) {
                            failures.add(testCase + " " + name + ": " + err + " > " + limit);
                        }
                    }
                    List<Forecast> fc = pipeline.finish(prep, s.head(), h);
                    double fe = TotoGolden.forecastError(fc, g.get("quantiles"));
                    if (!(fe <= TOLERANCE)) {
                        failures.add(testCase + " forecast: " + fe);
                    }
                    System.out.printf("%s rows=%d patches=%d worst %.2e (%s) forecast %.2e  %d ms%n", testCase,
                            prep.input().rows(), prep.input().patches(), worst, worstStage, fe, ms);
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    /** The forward's {@code [rows, K, patch, q]} in the golden's {@code [q, rows, K, patch]}. */
    static float[] headLayout(float[] head, int rows, int k, TotoConfig c) {
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
