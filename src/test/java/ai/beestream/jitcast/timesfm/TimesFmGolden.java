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
 * Reads the goldens of {@code scripts/timesfm/dump_golden.py}: {@code -Dtimesfm.checkpoint} (a directory
 * holding {@code model.safetensors} and {@code config.json}) and {@code -Dtimesfm.golden}. Tests skip
 * without them.
 */
public final class TimesFmGolden {

    private TimesFmGolden() {}

    public static Path checkpoint() {
        return required("timesfm.checkpoint");
    }

    public static Path goldenDir() {
        return required("timesfm.golden");
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

    /** The number of decode calls recorded for a case. */
    public static int calls(SafeTensors g) {
        int k = 0;
        while (g.contains("call" + k + ".pf")) {
            k++;
        }
        return k;
    }

    /** The prefill of decode call {@code k}: its normalized patches and masks. */
    public static PatchForward.Batch prefill(SafeTensors g, int k, int p) {
        Tensor normed = g.get("call" + k + ".normed");
        int rows = normed.dim(0), patches = normed.dim(1);
        return new PatchForward.Batch(rows, patches, normed.toFloatArray(), g.get("call" + k + ".in.masks").toFloatArray(), true);
    }

    /** {@code [rows][patches]}: whether a patch is observed (its last step is not padding). */
    public static boolean[][] observed(PatchForward.Batch b, int p) {
        boolean[][] o = new boolean[b.rows()][b.patches()];
        for (int r = 0; r < b.rows(); r++) {
            for (int t = 0; t < b.patches(); t++) {
                o[r][t] = b.masks()[((r * b.patches() + t) + 1) * p - 1] < 0.5f;
            }
        }
        return o;
    }

    /** The case's inputs as series: each golden row without its leading NaN padding. */
    public static List<TimeSeries> series(SafeTensors g, Map<String, Object> info) {
        Tensor ctx = g.get("in.context");
        int rows = ctx.dim(0), len = ctx.dim(1);
        float[] all = ctx.toFloatArray();
        boolean ragged = Boolean.TRUE.equals(info.get("ragged"));
        List<TimeSeries> out = new ArrayList<>();
        for (int r = 0; r < rows; r++) {
            int start = 0;
            if (ragged) {
                while (start < len && Float.isNaN(all[r * len + start])) {
                    start++;
                }
            }
            float[] v = new float[len - start];
            System.arraycopy(all, r * len + start, v, 0, v.length);
            out.add(TimeSeries.univariate(v));
        }
        return out;
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

    /** The {@code width}-wide vectors of the observed patches of a {@code [rows · patches · width]} activation. */
    public static float[] observedPatches(float[] values, boolean[][] observed, int width) {
        int rows = observed.length, patches = observed[0].length, count = 0;
        for (boolean[] row : observed) {
            for (boolean b : row) {
                count += b ? 1 : 0;
            }
        }
        float[] out = new float[count * width];
        int o = 0;
        for (int i = 0; i < rows; i++) {
            for (int t = 0; t < patches; t++) {
                if (observed[i][t]) {
                    System.arraycopy(values, (i * patches + t) * width, out, o, width);
                    o += width;
                }
            }
        }
        return out;
    }

    /** The golden noise floor of a stage, or 0 when not recorded. */
    public static double noise(SafeTensors g, String key) {
        return g.contains(key) ? g.get(key).getFloat(0) : 0;
    }
}
