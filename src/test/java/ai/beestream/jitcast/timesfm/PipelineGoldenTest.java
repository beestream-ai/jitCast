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
package ai.beestream.jitcast.timesfm;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.timesfm.cpu.CpuTimesFm;
import org.junit.Test;

/**
 * The host pipeline with the CPU forward pass against the reference: every decode call's assembled inputs
 * (the batch, the masks), its decoded outputs (the prefill's last patch, the continuous quantile head and
 * every fed-back step) and the final forecasts, for every golden case and its {@code ForecastConfig}.
 */
public class PipelineGoldenTest {

    private static final double INPUT_TOLERANCE = 1e-6;
    private static final double TOLERANCE = 1e-4;

    @Test
    public void forecastsMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TimesFmWeights weights = TimesFmWeights.load(TimesFmGolden.checkpoint())) {
            TimesFmConfig c = weights.config();
            CpuTimesFm cpu = new CpuTimesFm(weights);
            for (String testCase : TimesFmGolden.cases()) {
                Map<String, Object> info = TimesFmGolden.caseInfo(testCase);
                @SuppressWarnings("unchecked")
                Map<String, Object> fc = (Map<String, Object>) info.get("config");
                TimesFmPipeline pipeline = new TimesFmPipeline(c, cpu, options(c, fc), 1024, Long.MAX_VALUE);
                int h = ((Number) info.get("H")).intValue();
                try (SafeTensors g = TimesFmGolden.open(testCase)) {
                    List<TimeSeries> series = TimesFmGolden.series(g, info);
                    List<float[]> rows = new ArrayList<>();
                    for (TimeSeries ts : series) {
                        rows.add(ts.target()[0]);
                    }
                    TimesFmPipeline.Assembled a = pipeline.assemble(rows);
                    String worst = "";
                    double worstErr = 0;
                    for (int k = 0; k < TimesFmGolden.calls(g); k++) {
                        float[][] in = a.inputs();
                        if (k == 1) {
                            in = negate(in);
                        }
                        double inErr = TimesFmGolden.relativeMaxError(flat(in), g.get("call" + k + ".in.inputs").toFloatArray());
                        double maskErr = TimesFmGolden.relativeMaxError(flat(a.masks()), g.get("call" + k + ".in.masks").toFloatArray());
                        if (inErr > INPUT_TOLERANCE || maskErr > 0) {
                            failures.add(testCase + " call" + k + " inputs " + inErr + " masks " + maskErr);
                        }
                        int decodeLen = ((Number) ((List<?>) info.get("decode_horizons")).get(k)).intValue();
                        int steps = (decodeLen - 1) / c.horizonLength();
                        TimesFmPipeline.Decoded d = pipeline.decode(flatRows(g.get("call" + k + ".in.inputs")),
                                flatRows(g.get("call" + k + ".in.masks")), steps);
                        double e = TimesFmGolden.relativeMaxError(flat3(d.last()), lastPatch(g.get("call" + k + ".pf")));
                        double es = TimesFmGolden.relativeMaxError(flat3(d.spread()), g.get("call" + k + ".spread").toFloatArray());
                        double ea = steps == 0 ? 0 : TimesFmGolden.relativeMaxError(flat4(d.ar()), g.get("call" + k + ".ar").toFloatArray());
                        for (Object[] st : new Object[][] {{"pf", e}, {"spread", es}, {"ar", ea}}) {
                            double err = (Double) st[1];
                            if (err > worstErr) {
                                worstErr = err;
                                worst = "call" + k + "." + st[0];
                            }
                            if (!(err <= TOLERANCE)) {
                                failures.add(testCase + " call" + k + " " + st[0] + ": " + err);
                            }
                        }
                    }
                    List<Forecast> out = pipeline.forecast(series, h);
                    double fe = forecastError(out, g.get("quantiles"));
                    if (!(fe <= TOLERANCE)) {
                        failures.add(testCase + " forecast: " + fe);
                    }
                    System.out.printf("%s: decode worst %.2e (%s), forecast %.2e%n", testCase, worstErr, worst, fe);
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    static TimesFmPipeline.Options options(TimesFmConfig c, Map<String, Object> fc) {
        return new TimesFmPipeline.Options(((Number) fc.get("max_context")).intValue(), Boolean.TRUE.equals(fc.get("normalize_inputs")),
                Boolean.TRUE.equals(fc.get("use_continuous_quantile_head")), Boolean.TRUE.equals(fc.get("force_flip_invariance")),
                Boolean.TRUE.equals(fc.get("infer_is_positive")), Boolean.TRUE.equals(fc.get("fix_quantile_crossing")));
    }

    /** The golden {@code (B, H, 10)} quantiles (index 0 the mean) against the forecasts' 9 levels. */
    static double forecastError(List<Forecast> fc, Tensor golden) {
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

    private static float[] lastPatch(Tensor pf) {
        int b = pf.dim(0), n = pf.dim(1), per = (int) (pf.elements() / (b * n));
        float[] all = pf.toFloatArray(), out = new float[b * per];
        for (int r = 0; r < b; r++) {
            System.arraycopy(all, (r * n + n - 1) * per, out, r * per, per);
        }
        return out;
    }

    private static float[][] negate(float[][] x) {
        float[][] y = new float[x.length][];
        for (int r = 0; r < x.length; r++) {
            y[r] = x[r].clone();
            for (int i = 0; i < y[r].length; i++) {
                y[r][i] = -y[r][i];
            }
        }
        return y;
    }

    private static float[][] flatRows(Tensor t) {
        int r = t.dim(0), n = (int) (t.elements() / r);
        float[] all = t.toFloatArray();
        float[][] out = new float[r][n];
        for (int i = 0; i < r; i++) {
            System.arraycopy(all, i * n, out[i], 0, n);
        }
        return out;
    }

    private static float[] flat(float[][] x) {
        int n = x[0].length;
        float[] out = new float[x.length * n];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, out, i * n, n);
        }
        return out;
    }

    private static float[] flat3(float[][][] x) {
        List<Float> l = new ArrayList<>();
        for (float[][] a : x) {
            for (float[] b : a) {
                for (float v : b) {
                    l.add(v);
                }
            }
        }
        float[] out = new float[l.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = l.get(i);
        }
        return out;
    }

    private static float[] flat4(float[][][][] x) {
        List<float[]> parts = new ArrayList<>();
        int total = 0;
        for (float[][][] a : x) {
            float[] f = flat3(a);
            parts.add(f);
            total += f.length;
        }
        float[] out = new float[total];
        int o = 0;
        for (float[] f : parts) {
            System.arraycopy(f, 0, out, o, f.length);
            o += f.length;
        }
        return out;
    }
}
