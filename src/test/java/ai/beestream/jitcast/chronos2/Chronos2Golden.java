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
package ai.beestream.jitcast.chronos2;

import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.util.Json;

/**
 * Locates a Chronos-2 checkpoint directory ({@code -Dchronos2.checkpoint}) and a golden directory
 * ({@code -Dchronos2.golden}) written by {@code scripts/chronos2/dump_golden.py} from the same
 * checkpoint, and reads the golden files. Tests that need them are skipped, not failed, when they
 * are not configured.
 */
public final class Chronos2Golden {

    private Chronos2Golden() {}

    public static Path checkpoint() {
        return required("chronos2.checkpoint");
    }

    public static Path goldenDir() {
        return required("chronos2.golden");
    }

    private static Path required(String property) {
        String v = System.getProperty(property);
        assumeTrue("-D" + property + " not set", v != null && !v.isBlank() && !v.startsWith("${"));
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

    public static int horizon(Map<String, Object> info) {
        return ((Number) info.get("H")).intValue();
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

    /** The k-th recorded forward call's arguments. */
    public static ModelInput input(SafeTensors g, int k, int patch) {
        String p = "call" + k + ".in.";
        float[][] context = rows(g.get(p + "context"));
        int[] group = ints(g.get(p + "group_ids").toFloatArray());
        float[][] future = g.contains(p + "future_covariates") ? rows(g.get(p + "future_covariates")) : null;
        int h = g.get("call" + k + ".out").dim(2) / patch;
        return new ModelInput(context, group, future, h);
    }

    /** The k-th call's {@code [rows][tokens]} valid-key mask. */
    public static boolean[][] live(SafeTensors g, int k) {
        float[][] m = rows(g.get("call" + k + ".attention_mask"));
        boolean[][] b = new boolean[m.length][m[0].length];
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                b[i][j] = m[i][j] != 0f;
            }
        }
        return b;
    }

    /** The predict() options of a case. */
    public static Chronos2Pipeline.Options options(Chronos2Config c, Map<String, Object> info) {
        Chronos2Pipeline.Options o = Chronos2Pipeline.Options.defaults(c);
        if (info.get("kwargs") instanceof Map<?, ?> m && m.get("cross_learning") instanceof Boolean b) {
            o = o.withCrossLearning(b);
        }
        return o;
    }

    /** The golden inputs as series; a ragged case's series without their {@code NaN} padding. */
    public static List<TimeSeries> series(SafeTensors g, Map<String, Object> info) {
        boolean ragged = Boolean.TRUE.equals(info.get("ragged"));
        float[][][] ctx = cube(g.get("in.context"));
        float[][][] past = g.contains("in.past_covariates") ? cube(g.get("in.past_covariates")) : null;
        float[][][] known = g.contains("in.future_covariates") ? cube(g.get("in.future_covariates")) : null;
        List<TimeSeries> out = new ArrayList<>();
        for (int b = 0; b < ctx.length; b++) {
            int start = 0;
            if (ragged) {
                while (start < ctx[b][0].length && allNaN(ctx[b], start)) {
                    start++;
                }
            }
            out.add(new TimeSeries(from(ctx[b], start), past == null ? null : from(past[b], start),
                    known == null ? null : from(known[b], start)));
        }
        return out;
    }

    private static boolean allNaN(float[][] rows, int t) {
        for (float[] r : rows) {
            if (!Float.isNaN(r[t])) {
                return false;
            }
        }
        return true;
    }

    private static float[][] from(float[][] rows, int start) {
        float[][] r = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            r[i] = Arrays.copyOfRange(rows[i], start, rows[i].length);
        }
        return r;
    }

    /** Relative max error of the forecasts against the golden {@code (B, V, Q, H)} quantiles. */
    public static double forecastError(List<Forecast> fc, Tensor golden) {
        int bs = golden.dim(0), vs = golden.dim(1), qs = golden.dim(2), hs = golden.dim(3);
        float[] exp = golden.toFloatArray();
        float[] act = new float[exp.length];
        for (int b = 0; b < bs; b++) {
            for (int v = 0; v < vs; v++) {
                for (int q = 0; q < qs; q++) {
                    System.arraycopy(fc.get(b).values()[v][q], 0, act, ((b * vs + v) * qs + q) * hs, hs);
                }
            }
        }
        return relativeMaxError(act, exp);
    }

    public static float[][] rows(Tensor t) {
        int r = t.dim(0);
        int c = (int) (t.elements() / r);
        float[] flat = t.toFloatArray();
        float[][] out = new float[r][c];
        for (int i = 0; i < r; i++) {
            System.arraycopy(flat, i * c, out[i], 0, c);
        }
        return out;
    }

    static float[][][] cube(Tensor t) {
        int a = t.dim(0), b = t.dim(1), n = t.dim(2);
        float[] flat = t.toFloatArray();
        float[][][] r = new float[a][b][n];
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                System.arraycopy(flat, (i * b + j) * n, r[i][j], 0, n);
            }
        }
        return r;
    }

    static int[] ints(float[] v) {
        int[] r = new int[v.length];
        for (int i = 0; i < v.length; i++) {
            r[i] = Math.round(v[i]);
        }
        return r;
    }

    /** {@code [rows][Q][steps]} flattened row-major, as the golden {@code out} holds it. */
    public static float[] flat(float[][][] v) {
        int q = v[0].length, s = v[0][0].length;
        float[] f = new float[v.length * q * s];
        for (int i = 0; i < v.length; i++) {
            for (int k = 0; k < q; k++) {
                System.arraycopy(v[i][k], 0, f, (i * q + k) * s, s);
            }
        }
        return f;
    }

    /** {@code max|a − e| / max(max|e|, 1e-6)}, or infinity when a NaN appears where none is expected. */
    public static double relativeMaxError(float[] actual, float[] expected) {
        if (actual.length != expected.length) {
            throw new AssertionError("length " + actual.length + " != " + expected.length);
        }
        double diff = 0, scale = 1e-6;
        for (int i = 0; i < expected.length; i++) {
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

    /** The {@code width}-wide vectors of the valid tokens of a {@code [rows · tokens · width]} activation. */
    public static float[] liveTokens(float[] values, boolean[][] live, int width) {
        int rows = live.length, tokens = live[0].length;
        int count = 0;
        for (boolean[] row : live) {
            for (boolean b : row) {
                count += b ? 1 : 0;
            }
        }
        float[] out = new float[count * width];
        int o = 0;
        for (int i = 0; i < rows; i++) {
            for (int t = 0; t < tokens; t++) {
                if (live[i][t]) {
                    System.arraycopy(values, (i * tokens + t) * width, out, o, width);
                    o += width;
                }
            }
        }
        return out;
    }
}
