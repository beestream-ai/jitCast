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
package ai.beestream.jitrex.tornado;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitrex.GoldenData;
import ai.beestream.jitrex.api.Forecast;
import ai.beestream.jitrex.api.TimeSeries;
import ai.beestream.jitrex.io.SafeTensors;
import ai.beestream.jitrex.model.ModelBatch;
import org.junit.Test;

/**
 * The TornadoVM forward pass and engine against the reference, on every golden case (univariate,
 * multi-target, past and future covariates, mixed batches). Runs under -Paccel-tests; pick the device with
 * {@code -Dtirex.tornado.device=<driver>:<device>}.
 */
public class TornadoForwardAccelTest {

    // FP32 configurations: within 1e-4 of the FP32 reference (they are ~1e-5 from it). Configurations
    // with FP16/TF32 anywhere: per case, no further from the reference than NXAI's own GPU path is
    // (GoldenData.nxaiGpuError), since NXAI accepts its GPU path by its benchmark scores, not by an
    // error bound.
    private static final double FP32_TOLERANCE = 1e-4;

    @Test
    public void forwardAndForecastMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        int checked = 0;
        boolean reduced = ai.beestream.jitrex.tornado.TornadoForward.reducedPrecision();
        if (reduced) {
            GoldenData.gpuGoldenDir();
        }
        try (TornadoTiRex2Engine engine = TornadoTiRex2Engine.load(GoldenData.weights())) {
            for (String testCase : GoldenData.cases()) {
                try (SafeTensors g = GoldenData.open(testCase)) {
                    ModelBatch batch = GoldenData.batch(g, 0);
                    checked++;
                    long t0 = System.nanoTime();
                    float[][][] out = engine.forwardPass().forward(batch);
                    long first = (System.nanoTime() - t0) / 1_000_000;
                    t0 = System.nanoTime();
                    out = engine.forwardPass().forward(batch);
                    double again = (System.nanoTime() - t0) / 1e6;
                    double err = GoldenData.relativeMaxError(flatten(out), g.get("pass0.model.out").toFloatArray());

                    Map<String, Object> info = GoldenData.caseInfo(testCase);
                    int h = ((Number) info.get("H")).intValue();
                    @SuppressWarnings("unchecked")
                    Map<String, Object> kw = (Map<String, Object>) info.get("kwargs");
                    boolean signFlip = kw.containsKey("tta_sign_flip") ? Boolean.TRUE.equals(kw.get("tta_sign_flip"))
                            : engine.pipeline().signFlip();
                    List<TimeSeries> in = GoldenData.inputs(g);
                    List<Forecast> f = engine.pipeline().withSignFlip(signFlip).forecast(in, h);
                    double ferr = 0;
                    for (int s = 0; s < in.size(); s++) {
                        ferr = Math.max(ferr, GoldenData.relativeMaxError(flatten(f.get(s).values()),
                                g.get("forecast." + s).toFloatArray()));
                    }
                    double[] limit = reduced ? GoldenData.nxaiGpuError(testCase) : new double[] {FP32_TOLERANCE, FP32_TOLERANCE};
                    System.out.printf("%-26s rows=%2d  first %6d ms  again %8.2f ms  model.out %.2e  forecast %.2e  (limits %.2e %.2e)%n",
                            testCase, batch.rows(), first, again, err, ferr, limit[0], limit[1]);
                    if (err > limit[0] || ferr > limit[1]) {
                        failures.add(testCase + ": model.out " + err + ", forecast " + ferr);
                    }
                }
            }
        }
        assertTrue("no case in scope", checked > 0);
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    static float[] flatten(float[][][] v) {
        float[] flat = new float[v.length * v[0].length * v[0][0].length];
        int i = 0;
        for (float[][] a : v) {
            for (float[] b : a) {
                System.arraycopy(b, 0, flat, i, b.length);
                i += b.length;
            }
        }
        return flat;
    }
}
