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
package ai.beestream.jitcast.tirex2;

import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.util.Json;

/**
 * Locates the converted weights ({@code -Dtirex.weights}) and a golden directory
 * ({@code -Dtirex.golden}) written by {@code scripts/dump_golden.py}, and optionally NXAI's CUDA
 * goldens ({@code -Dtirex.golden.gpu}, {@code dump_golden.py --device cuda}). Tests that need them
 * are skipped, not failed, when they are not configured.
 */
public final class GoldenData {

    private GoldenData() {}

    public static Path weights() {
        return required("tirex.weights");
    }

    public static Path goldenDir() {
        return required("tirex.golden");
    }

    /** NXAI's CUDA goldens: the same cases as {@link #goldenDir()}, run on NXAI's own GPU path. */
    public static Path gpuGoldenDir() {
        return required("tirex.golden.gpu");
    }

    /**
     * How far NXAI's own GPU path is from its FP32 reference on a case, as {@code {model output,
     * forecasts}} in {@link #relativeMaxError}: the limits for jitcast's reduced-precision
     * configurations, which may be no further from FP32 than NXAI is. Both golden sets must come from
     * the same inputs, so record them on the same machine: the generated inputs differ in their last
     * bits between CPU architectures.
     */
    public static double[] nxaiGpuError(String testCase) throws IOException {
        try (SafeTensors cpu = open(testCase); SafeTensors gpu = SafeTensors.open(gpuGoldenDir().resolve(testCase + ".safetensors"))) {
            for (int i = 0; cpu.contains("in." + i + ".target"); i++) {
                for (String part : List.of("target", "past_covariates", "future_covariates")) {
                    String name = "in." + i + "." + part;
                    if (cpu.contains(name) && !java.util.Arrays.equals(cpu.get(name).toFloatArray(), gpu.get(name).toFloatArray())) {
                        throw new AssertionError(testCase + ": the CPU and GPU goldens have different inputs (" + name + ")");
                    }
                }
            }
            double model = relativeMaxError(gpu.get("pass0.model.out").toFloatArray(), cpu.get("pass0.model.out").toFloatArray());
            double forecast = 0;
            for (int i = 0; cpu.contains("forecast." + i); i++) {
                forecast = Math.max(forecast, relativeMaxError(gpu.get("forecast." + i).toFloatArray(), cpu.get("forecast." + i).toFloatArray()));
            }
            return new double[] {model, forecast};
        }
    }

    /** Case names from {@code manifest.json}, in dump order. */
    @SuppressWarnings("unchecked")
    public static List<String> cases() throws IOException {
        Map<String, Object> manifest = Json.parseObject(Files.readString(goldenDir().resolve("manifest.json")));
        return List.copyOf(((Map<String, Object>) manifest.get("cases")).keySet());
    }

    public static SafeTensors open(String testCase) throws IOException {
        return SafeTensors.open(goldenDir().resolve(testCase + ".safetensors"));
    }

    /** Number of forward passes recorded for a case (2 with sign-flip TTA). */
    @SuppressWarnings("unchecked")
    public static int passes(String testCase) throws IOException {
        Map<String, Object> manifest = Json.parseObject(Files.readString(goldenDir().resolve("manifest.json")));
        Map<String, Object> info = (Map<String, Object>) ((Map<String, Object>) manifest.get("cases")).get(testCase);
        return ((Number) info.get("passes")).intValue();
    }

    /** The {@code forecast()} inputs of a case: {@code in.<i>.target / past_covariates / future_covariates}. */
    public static List<TimeSeries> inputs(SafeTensors g) {
        List<TimeSeries> out = new java.util.ArrayList<>();
        for (int i = 0; g.contains("in." + i + ".target"); i++) {
            out.add(new TimeSeries(matrix(g, "in." + i + ".target"), matrix(g, "in." + i + ".past_covariates"),
                    matrix(g, "in." + i + ".future_covariates")));
        }
        return out;
    }

    /** A 2-D F32 tensor as rows, or null if absent. */
    public static float[][] matrix(SafeTensors g, String name) {
        if (!g.contains(name)) {
            return null;
        }
        Tensor t = g.get(name);
        float[] flat = t.toFloatArray();
        int cols = t.dim(-1), rows = (int) (t.elements() / cols);
        float[][] m = new float[rows][];
        for (int r = 0; r < rows; r++) {
            m[r] = java.util.Arrays.copyOfRange(flat, r * cols, (r + 1) * cols);
        }
        return m;
    }

    /** A case's forecast() horizon and sign-flip flag from manifest.json. */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> caseInfo(String testCase) throws IOException {
        Map<String, Object> manifest = Json.parseObject(Files.readString(goldenDir().resolve("manifest.json")));
        return (Map<String, Object>) ((Map<String, Object>) manifest.get("cases")).get(testCase);
    }

    /** The batch {@code TiRex2.forward()} received in pass {@code pass}. */
    public static ModelBatch batch(SafeTensors g, int pass) {
        Tensor x = g.get("pass" + pass + ".batch.x");
        float[] flat = x.toFloatArray();
        int rows = x.dim(0), len = x.dim(1);
        float[][] rowsX = new float[rows][];
        for (int r = 0; r < rows; r++) {
            rowsX[r] = java.util.Arrays.copyOfRange(flat, r * len, (r + 1) * len);
        }
        float[] group = g.get("pass" + pass + ".batch.group_vector").toFloatArray();
        float[] targetF = g.get("pass" + pass + ".batch.target_mask").toFloatArray();
        boolean[] target = new boolean[rows];
        for (int r = 0; r < rows; r++) {
            target[r] = targetF[r] != 0f;
        }
        return new ModelBatch(rowsX, group, target);
    }

    /** {@code [a][b][c]} → {@code [b][a][c]}, flattened (the variate mixer's golden layout is [L, R, D]). */
    public static float[] swap01(float[] x, int a, int b, int c) {
        float[] y = new float[x.length];
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                System.arraycopy(x, (i * b + j) * c, y, (j * a + i) * c, c);
            }
        }
        return y;
    }

    private static Path required(String property) {
        String value = System.getProperty(property, "");
        assumeTrue("-D" + property + " not set", !value.isBlank());
        Path path = Path.of(value);
        assumeTrue(path + " does not exist", Files.exists(path));
        return path;
    }

    /** max|a-b| / max(max|b|, 1e-6): one number that is comparable across stages. */
    public static double relativeMaxError(float[] actual, float[] expected) {
        if (actual.length != expected.length) {
            throw new AssertionError("length " + actual.length + " != " + expected.length);
        }
        double diff = 0, scale = 1e-6;
        for (int i = 0; i < actual.length; i++) {
            diff = Math.max(diff, Math.abs(actual[i] - expected[i]));
            scale = Math.max(scale, Math.abs(expected[i]));
        }
        return diff / scale;
    }
}
