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
package ai.beestream.jitcast.toto;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.toto.cpu.CpuToto;
import org.junit.Test;

/**
 * The whole CPU pipeline against the reference's forecasts, from the raw series: imputation, trimming,
 * padding (to whole patches only, where the reference pads to its full window), the scaler and the
 * real-space steps. Also checks the scaler's statistics against the reference's.
 */
public class PipelineGoldenTest {

    private static final double TOLERANCE = 1e-4;

    @Test
    public void forecastsMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TotoWeights weights = TotoWeights.load(TotoGolden.checkpoint())) {
            TotoConfig c = weights.config();
            CpuToto cpu = new CpuToto(weights);
            for (String testCase : TotoGolden.cases()) {
                Map<String, Object> info = TotoGolden.caseInfo(testCase);
                int h = TotoGolden.intOf(info, "H");
                TotoPipeline pipeline = new TotoPipeline(c, cpu,
                        TotoPipeline.Options.defaults().withMaxContext(TotoGolden.intOf(info, "context_length")), 3);
                try (SafeTensors g = TotoGolden.open(testCase)) {
                    long t0 = System.nanoTime();
                    List<Forecast> fc = pipeline.forecast(TotoGolden.series(g), h);
                    long ms = (System.nanoTime() - t0) / 1_000_000;
                    double fe = TotoGolden.forecastError(fc, g.get("quantiles"));
                    double se = scalerError(g, c.patchSize());
                    if (!(fe <= TOLERANCE)) {
                        failures.add(testCase + " forecast: " + fe);
                    }
                    if (!(se <= 1e-6)) {
                        failures.add(testCase + " scaler: " + se);
                    }
                    System.out.printf("%s forecast %.2e scaler %.2e  %d ms%n", testCase, fe, se, ms);
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    /** The scaler on the reference's own inputs against its recorded statistics. */
    private static double scalerError(SafeTensors g, int p) {
        Tensor loc = g.get("loc");
        int b = loc.dim(0), v = loc.dim(1), len = loc.dim(2);
        float[] target = new float[b * v * len];
        boolean[] obs = new boolean[target.length];
        fill(g.get("in.target"), g.get("in.observed"), g.contains("in.past") ? g.get("in.past") : null,
                g.contains("in.past") ? g.get("in.past_observed") : null, target, obs, b, v, len, p);
        float[] myLoc = new float[target.length], myScale = new float[target.length];
        for (int r = 0; r < b * v; r++) {
            TotoPipeline.scaler(target, obs, r * len, len, p, myLoc, myScale);
        }
        return Math.max(TotoGolden.relativeMaxError(myLoc, loc.toFloatArray()),
                TotoGolden.relativeMaxError(myScale, g.get("scale").toFloatArray()));
    }

    private static void fill(Tensor t, Tensor o, Tensor pt, Tensor po, float[] target, boolean[] obs, int b, int v, int len, int p) {
        int vt = t.dim(1), c = t.dim(2);
        assertEquals(v, vt + (pt == null ? 0 : pt.dim(1)));
        float[] tv = t.toFloatArray(), ov = o.toFloatArray();
        float[] pv = pt == null ? null : pt.toFloatArray(), pov = po == null ? null : po.toFloatArray();
        for (int i = 0; i < b; i++) {
            for (int r = 0; r < v; r++) {
                int dst = (i * v + r) * len;
                for (int k = 0; k < c; k++) {
                    boolean isTarget = r < vt;
                    int src = isTarget ? (i * vt + r) * c + k : (i * (v - vt) + r - vt) * c + k;
                    target[dst + k] = isTarget ? tv[src] : pv[src];
                    obs[dst + k] = (isTarget ? ov[src] : pov[src]) > 0.5f || k >= c - p;
                }
            }
        }
    }
}
