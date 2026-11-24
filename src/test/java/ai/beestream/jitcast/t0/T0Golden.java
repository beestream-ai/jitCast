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
package ai.beestream.jitcast.t0;

import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.util.Json;

/**
 * Reads the goldens of {@code scripts/t0/dump_golden.py}: {@code -Dt0.checkpoint} (a directory holding
 * {@code model.safetensors} and {@code config.json}) and {@code -Dt0.golden}. Tests skip without them.
 */
public final class T0Golden {

    private T0Golden() {}

    public static Path checkpoint() {
        return required("t0.checkpoint");
    }

    public static Path goldenDir() {
        return required("t0.golden");
    }

    private static Path required(String property) {
        String v = System.getProperty(property);
        assumeTrue("-D" + property + " not set", v != null && !v.isBlank());
        Path p = Path.of(v);
        assumeTrue(p + " does not exist", Files.exists(p));
        return p;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> manifest() throws IOException {
        return (Map<String, Object>) Json.parseObject(Files.readString(goldenDir().resolve("manifest.json"))).get("cases");
    }

    public static List<String> cases() throws IOException {
        return List.copyOf(manifest().keySet());
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> caseInfo(String testCase) throws IOException {
        return (Map<String, Object>) manifest().get(testCase);
    }

    public static SafeTensors open(String testCase) throws IOException {
        return SafeTensors.open(goldenDir().resolve(testCase + ".safetensors"));
    }

    /** The number of forward calls recorded for a case. */
    public static int calls(SafeTensors g) {
        int k = 0;
        while (g.contains("call" + k + ".out")) {
            k++;
        }
        return k;
    }

    /** The window of forward call {@code k}, as the reference passed it. */
    public static T0Batch window(SafeTensors g, int k) {
        Tensor v = g.get("call" + k + ".in.variates");
        int rows = v.dim(0), width = v.dim(1);
        float[] mask = g.get("call" + k + ".in.mask").toFloatArray(), group = g.get("call" + k + ".in.group").toFloatArray(),
                type = g.get("call" + k + ".in.type").toFloatArray();
        T0Batch b = T0Batch.empty(rows, width);
        System.arraycopy(v.toFloatArray(), 0, b.values(), 0, rows * width);
        for (int i = 0; i < rows * width; i++) {
            b.mask()[i] = (byte) mask[i];
            b.group()[i] = (int) group[i];
            b.type()[i] = (int) type[i];
        }
        return b;
    }

    /** The case's inputs as jitcast series. */
    public static List<TimeSeries> series(SafeTensors g, Map<String, Object> info) {
        int n = ((Number) info.get("series")).intValue();
        List<TimeSeries> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            out.add(new TimeSeries(rows(g, "s" + i + ".target"), rows(g, "s" + i + ".past"), rows(g, "s" + i + ".future")));
        }
        return out;
    }

    private static float[][] rows(SafeTensors g, String key) {
        if (!g.contains(key)) {
            return null;
        }
        Tensor t = g.get(key);
        int r = t.dim(0), w = t.dim(1);
        float[] all = t.toFloatArray();
        float[][] out = new float[r][w];
        for (int i = 0; i < r; i++) {
            System.arraycopy(all, i * w, out[i], 0, w);
        }
        return out;
    }

    /** The pipeline options of a case, as the reference's predict() keywords. */
    @SuppressWarnings("unchecked")
    public static T0Pipeline.Options options(Map<String, Object> info) {
        Map<String, Object> o = (Map<String, Object>) info.get("options");
        List<?> q = (List<?>) o.get("quantile_levels");
        double[] levels = new double[q.size()];
        for (int i = 0; i < levels.length; i++) {
            levels[i] = ((Number) q.get(i)).doubleValue();
        }
        return new T0Pipeline.Options(levels, Boolean.TRUE.equals(o.get("trim_leading_missing")),
                Boolean.TRUE.equals(o.get("symmetric_averaging")), Boolean.TRUE.equals(o.get("clip_nonnegative")),
                Boolean.TRUE.equals(o.get("snap_integers")), Boolean.TRUE.equals(o.get("drop_constant_covariates")));
    }

    /** {@code max|a − e| / max(max|e|, 1e-6)} over the cells {@code keep} selects (all when null). */
    public static double relativeMaxError(float[] actual, float[] expected, boolean[] keep) {
        if (actual.length != expected.length) {
            throw new AssertionError("length " + actual.length + " != " + expected.length);
        }
        double diff = 0, scale = 1e-6;
        for (int i = 0; i < expected.length; i++) {
            if (keep != null && !keep[i]) {
                continue;
            }
            if (Float.isNaN(actual[i]) != Float.isNaN(expected[i])) {
                return Double.POSITIVE_INFINITY;
            }
            if (!Float.isNaN(expected[i])) {
                diff = Math.max(diff, Math.abs((double) actual[i] - expected[i]));
                scale = Math.max(scale, Math.abs(expected[i]));
            }
        }
        return diff / scale;
    }

    /** {@code keep} widened from one flag per group of {@code width} cells to one per cell. */
    public static boolean[] widen(boolean[] keep, int width) {
        boolean[] out = new boolean[keep.length * width];
        for (int i = 0; i < out.length; i++) {
            out[i] = keep[i / width];
        }
        return out;
    }

    /** The golden noise floor of a stage, or 0 when not recorded. */
    public static double noise(SafeTensors g, String key) {
        return g.contains(key) ? g.get(key).getFloat(0) : 0;
    }
}
