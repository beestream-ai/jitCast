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

import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.util.Json;

/**
 * Golden files of Datadog's Toto 2.0 reference (scripts/toto/dump_golden.py), located by
 * {@code -Dtoto.checkpoint} and {@code -Dtoto.golden}; tests skip without them.
 */
public final class TotoGolden {

    private TotoGolden() {}

    /** {@return the checkpoint directory} */
    public static Path checkpoint() {
        return required("toto.checkpoint");
    }

    /** {@return the golden directory} */
    public static Path goldenDir() {
        return required("toto.golden");
    }

    private static Path required(String property) {
        String v = System.getProperty(property);
        assumeTrue("-D" + property + " not set", v != null && !v.isBlank());
        Path p = Path.of(v);
        assumeTrue(p + " does not exist", Files.exists(p));
        return p;
    }

    /** {@return case name to its manifest entry} */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> manifest() throws IOException {
        return (Map<String, Object>) Json.parseObject(Files.readString(goldenDir().resolve("manifest.json"))).get("cases");
    }

    /** {@return the case names} */
    public static List<String> cases() throws IOException {
        return List.copyOf(manifest().keySet());
    }

    /** {@return one case's manifest entry} */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> caseInfo(String testCase) throws IOException {
        return (Map<String, Object>) manifest().get(testCase);
    }

    /** {@return one case's tensors} */
    public static SafeTensors open(String testCase) throws IOException {
        return SafeTensors.open(goldenDir().resolve(testCase + ".safetensors"));
    }

    /** {@return an integer of a manifest entry} */
    public static int intOf(Map<String, Object> info, String key) {
        return ((Number) info.get(key)).intValue();
    }

    /** {@return the case's raw series} */
    public static List<TimeSeries> series(SafeTensors g) {
        List<TimeSeries> out = new ArrayList<>();
        for (int i = 0; g.contains("series" + i + ".target"); i++) {
            float[][] past = g.contains("series" + i + ".past") ? rows(g.get("series" + i + ".past")) : null;
            out.add(new TimeSeries(rows(g.get("series" + i + ".target")), past, null));
        }
        return out;
    }

    /**
     * {@return the series exactly as the reference's forward saw them}: its trimmed, left-padded inputs,
     * NaN where unobserved, so that a pipeline with the case's context limit neither trims nor pads them
     */
    public static List<TimeSeries> assembled(SafeTensors g) {
        float[][][] target = split(g.get("in.target"), g.get("in.observed"));
        float[][][] past = g.contains("in.past") ? split(g.get("in.past"), g.get("in.past_observed")) : null;
        List<TimeSeries> out = new ArrayList<>();
        for (int b = 0; b < target.length; b++) {
            out.add(new TimeSeries(target[b], past == null ? null : past[b], null));
        }
        return out;
    }

    private static float[][][] split(Tensor values, Tensor observed) {
        int b = values.dim(0), v = values.dim(1), c = values.dim(2);
        float[] x = values.toFloatArray(), o = observed.toFloatArray();
        float[][][] out = new float[b][v][c];
        for (int i = 0; i < b; i++) {
            for (int r = 0; r < v; r++) {
                for (int t = 0; t < c; t++) {
                    int k = (i * v + r) * c + t;
                    out[i][r][t] = o[k] > 0.5f ? x[k] : Float.NaN;
                }
            }
        }
        return out;
    }

    private static float[][] rows(Tensor t) {
        int v = t.dim(0), n = t.dim(1);
        float[] all = t.toFloatArray();
        float[][] out = new float[v][n];
        for (int r = 0; r < v; r++) {
            System.arraycopy(all, r * n, out[r], 0, n);
        }
        return out;
    }

    /** {@return the float32 reference's distance from float64 at a stage, or 0 when not recorded} */
    public static double noise(SafeTensors g, String stage) {
        return g.contains("fp32err." + stage) ? g.get("fp32err." + stage).toFloatArray()[0] : 0;
    }

    /** {@return max |a − e| / max |e|} */
    public static double relativeMaxError(float[] a, float[] e) {
        if (a.length != e.length) {
            throw new IllegalArgumentException("length " + a.length + " vs " + e.length);
        }
        double diff = 0, ref = 0;
        for (int i = 0; i < a.length; i++) {
            double d = Math.abs((double) a[i] - e[i]);
            diff = Double.isNaN(d) ? Double.POSITIVE_INFINITY : Math.max(diff, d);
            ref = Math.max(ref, Math.abs(e[i]));
        }
        return diff / Math.max(ref, 1e-6);
    }

    /**
     * {@return the worst relative max error of any series' forecasts against the golden quantiles}, each series
     * measured against its own magnitude
     */
    public static double forecastError(List<Forecast> fc, Tensor golden) {
        int b = golden.dim(0), v = golden.dim(1), h = golden.dim(2), q = golden.dim(3);
        float[] g = golden.toFloatArray();
        double worst = 0;
        for (int s = 0; s < b; s++) {
            float[] exp = new float[v * q * h], act = new float[exp.length];
            for (int t = 0; t < v; t++) {
                for (int l = 0; l < q; l++) {
                    for (int i = 0; i < h; i++) {
                        exp[(t * q + l) * h + i] = g[((s * v + t) * h + i) * q + l];
                        act[(t * q + l) * h + i] = fc.get(s).values()[t][l][i];
                    }
                }
            }
            worst = Math.max(worst, relativeMaxError(act, exp));
        }
        return worst;
    }
}
