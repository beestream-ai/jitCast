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
package ai.beestream.jitrex.timestral;

import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import ai.beestream.jitrex.io.SafeTensors;
import ai.beestream.jitrex.io.Tensor;
import ai.beestream.jitrex.util.Json;

/**
 * Locates a Timestral checkpoint directory ({@code -Dtimestral.checkpoint}) and a golden directory
 * ({@code -Dtimestral.golden}) written by {@code scripts/timestral/dump_golden.py} from the same
 * checkpoint. Tests that need them are skipped, not failed, when they are not configured.
 */
public final class TimestralGolden {

    private TimestralGolden() {}

    public static Path checkpoint() {
        return required("timestral.checkpoint");
    }

    public static Path goldenDir() {
        return required("timestral.golden");
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

    public static SafeTensors open(String testCase) throws IOException {
        return SafeTensors.open(goldenDir().resolve(testCase + ".safetensors"));
    }

    /** The number of {@code predict_block} calls recorded for a case. */
    public static int calls(SafeTensors g) {
        int k = 0;
        while (g.contains("call" + k + ".out")) {
            k++;
        }
        return k;
    }

    /** The k-th recorded {@code predict_block} call's arguments. */
    public static BlockInput block(SafeTensors g, int k, int patch) {
        String p = "call" + k + ".in.";
        float[][] inputs = rows(g.get(p + "inputs"));
        boolean[][] observed = mask(rows(g.get(p + "observed")));
        int[] group = ints(g.get(p + "group_ids").toFloatArray());
        int[] role = g.contains(p + "role_ids") ? ints(g.get(p + "role_ids").toFloatArray()) : null;
        float[][] kf = g.contains(p + "known_future") ? rows(g.get(p + "known_future")) : null;
        boolean[][] kfm = g.contains(p + "known_future_mask") ? mask(rows(g.get(p + "known_future_mask"))) : null;
        int h = g.get("call" + k + ".out").dim(1) / patch;
        return new BlockInput(inputs, observed, group, h, role, kf, kfm);
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

    static boolean[][] mask(float[][] m) {
        boolean[][] b = new boolean[m.length][m[0].length];
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[i].length; j++) {
                b[i][j] = m[i][j] != 0f;
            }
        }
        return b;
    }

    static int[] ints(float[] v) {
        int[] r = new int[v.length];
        for (int i = 0; i < v.length; i++) {
            r[i] = Math.round(v[i]);
        }
        return r;
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

    /**
     * Rows {@code [R][from·width, to·width)} of a {@code [R][tokens·width]} activation, flattened:
     * the slice of every row's tokens {@code from .. to}.
     */
    public static float[] tokens(float[] values, int rows, int tokens, int width, int from, int to) {
        float[] out = new float[rows * (to - from) * width];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(values, (i * tokens + from) * width, out, i * (to - from) * width, (to - from) * width);
        }
        return out;
    }
}
