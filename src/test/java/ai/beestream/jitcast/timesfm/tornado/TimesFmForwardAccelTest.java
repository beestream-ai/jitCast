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
package ai.beestream.jitcast.timesfm.tornado;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.timesfm.PatchForward;
import ai.beestream.jitcast.timesfm.TimesFmConfig;
import ai.beestream.jitcast.timesfm.TimesFmGolden;
import ai.beestream.jitcast.timesfm.TimesFmPipeline;
import ai.beestream.jitcast.timesfm.TimesFmWeights;
import org.junit.Test;

/**
 * The TornadoVM forward pass against the reference: the heads of each recorded prefill's last patch, and
 * the whole pipeline's forecasts, for every golden case. FP32 GEMM modes must match within {@code 1e-4};
 * modes with FP16 or TF32 GEMM inputs within {@link #REDUCED}.
 */
public class TimesFmForwardAccelTest {

    private static final double TOLERANCE = 1e-4;
    private static final double REDUCED = 5e-2;

    @Test
    public void headsAndForecastsMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TimesFmWeights weights = TimesFmWeights.load(TimesFmGolden.checkpoint());
                TimesFmForward fwd = new TimesFmForward(weights.config(), TimesFmPacked.pack(weights), 4)) {
            TimesFmConfig c = weights.config();
            double limit = fwd.gemmMode().reducedPrecision() ? REDUCED : TOLERANCE;
            System.out.println("GEMM mode " + fwd.gemmMode() + ", limit " + limit);
            for (String testCase : TimesFmGolden.cases()) {
                Map<String, Object> info = TimesFmGolden.caseInfo(testCase);
                try (SafeTensors g = TimesFmGolden.open(testCase)) {
                    double worst = 0;
                    for (int k = 0; k < TimesFmGolden.calls(g); k++) {
                        PatchForward.Batch in = TimesFmGolden.prefill(g, k, c.patchLength());
                        long t0 = System.nanoTime();
                        PatchForward.Heads h = fwd.forward(in);
                        long ms = (System.nanoTime() - t0) / 1_000_000;
                        double ep = TimesFmGolden.relativeMaxError(h.point(), lastPatch(g.get("call" + k + ".head_point")));
                        double es = TimesFmGolden.relativeMaxError(h.spread(), g.get("call" + k + ".head_spread").toFloatArray());
                        worst = Math.max(worst, Math.max(ep, es));
                        if (!(ep <= limit && es <= limit)) {
                            failures.add(testCase + " call" + k + " heads: point " + ep + " spread " + es);
                        }
                        System.out.printf("%s call%d rows=%d patches=%d point %.2e spread %.2e  %d ms%n", testCase, k, in.rows(),
                                in.patches(), ep, es, ms);
                    }
                    @SuppressWarnings("unchecked")
                    Map<String, Object> fc = (Map<String, Object>) info.get("config");
                    TimesFmPipeline pipeline = new TimesFmPipeline(c, fwd, options(fc), 1024, Long.MAX_VALUE);
                    List<TimeSeries> series = TimesFmGolden.series(g, info);
                    int horizon = ((Number) info.get("H")).intValue();
                    long t0 = System.nanoTime();
                    List<Forecast> out = pipeline.forecast(series, horizon);
                    long ms = (System.nanoTime() - t0) / 1_000_000;
                    double fe = forecastError(out, g.get("quantiles"));
                    if (!(fe <= limit)) {
                        failures.add(testCase + " forecast: " + fe);
                    }
                    System.out.printf("%s forecast %.2e  %d ms%n", testCase, fe, ms);
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    private static TimesFmPipeline.Options options(Map<String, Object> fc) {
        return new TimesFmPipeline.Options(((Number) fc.get("max_context")).intValue(), Boolean.TRUE.equals(fc.get("normalize_inputs")),
                Boolean.TRUE.equals(fc.get("use_continuous_quantile_head")), Boolean.TRUE.equals(fc.get("force_flip_invariance")),
                Boolean.TRUE.equals(fc.get("infer_is_positive")), Boolean.TRUE.equals(fc.get("fix_quantile_crossing")));
    }

    private static float[] lastPatch(Tensor t) {
        int b = t.dim(0), n = t.dim(1), per = (int) (t.elements() / (b * n));
        float[] all = t.toFloatArray(), out = new float[b * per];
        for (int r = 0; r < b; r++) {
            System.arraycopy(all, (r * n + n - 1) * per, out, r * per, per);
        }
        return out;
    }

    private static double forecastError(List<Forecast> fc, Tensor golden) {
        int b = golden.dim(0), h = golden.dim(1), q = golden.dim(2);
        float[] g = golden.toFloatArray();
        float[] exp = new float[b * (q - 1) * h], act = new float[exp.length];
        for (int r = 0; r < b; r++) {
            for (int k = 1; k < q; k++) {
                for (int t = 0; t < h; t++) {
                    exp[(r * (q - 1) + k - 1) * h + t] = g[(r * h + t) * q + k];
                    act[(r * (q - 1) + k - 1) * h + t] = fc.get(r).values()[0][k - 1][t];
                }
            }
        }
        return TimesFmGolden.relativeMaxError(act, exp);
    }
}
