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

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.t0.cpu.CpuT0;

/**
 * The host pipeline with the CPU forward pass against the reference, for every golden case and its
 * {@code predict()} options: every forward window it builds (bit for bit: values, masks, groups and types,
 * through batching, trimming, dropped covariates, the negated series and the rollout's trajectories; the
 * forecasts a rollout feeds back match to the tolerance), each
 * window's predictions in data space, and the final forecasts. The batch holds every series of a case, as the
 * reference's did; {@link #batchingDoesNotChangeForecasts} checks that smaller batches give the same forecasts.
 */
public class PipelineGoldenTest {

    private static final double TOLERANCE = 1e-4;

    @Test
    public void forecastsMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (T0Weights weights = T0Weights.load(T0Golden.checkpoint())) {
            T0Config c = weights.config();
            CpuT0 cpu = new CpuT0(weights);
            for (String testCase : T0Golden.cases()) {
                Map<String, Object> info = T0Golden.caseInfo(testCase);
                int h = ((Number) info.get("H")).intValue();
                try (SafeTensors g = T0Golden.open(testCase)) {
                    List<T0Batch> windows = new ArrayList<>();
                    List<float[]> outs = new ArrayList<>();
                    T0Pipeline pipeline = new T0Pipeline(c, cpu, T0Golden.options(info), 1 << 20, Long.MAX_VALUE)
                            .observe((w, horizon, out) -> {
                                windows.add(w);
                                outs.add(out);
                            });
                    List<Forecast> fc = pipeline.forecast(T0Golden.series(g, info), h);
                    int calls = T0Golden.calls(g);
                    if (windows.size() != calls) {
                        failures.add(testCase + ": " + windows.size() + " forward calls, the reference made " + calls);
                    }
                    double worstOut = 0;
                    List<?> horizons = (List<?>) info.get("horizons");
                    for (int k = 0; k < Math.min(calls, windows.size()); k++) {
                        // A rollout's later windows carry the forecasts fed back, which match only to the tolerance.
                        boolean fedBack = k > 0 && ((Number) horizons.get(k - 1)).intValue() == T0Config.MAX_HORIZON;
                        String in = sameWindow(windows.get(k), T0Golden.window(g, k), fedBack ? TOLERANCE : 1e-7);
                        if (in != null) {
                            failures.add(testCase + " call" + k + " window: " + in);
                        }
                        double e = T0Golden.relativeMaxError(outs.get(k), g.get("call" + k + ".out").toFloatArray(), null);
                        worstOut = Math.max(worstOut, e);
                        if (!(e <= TOLERANCE)) {
                            failures.add(testCase + " call" + k + " out: " + e);
                        }
                    }
                    double fe = forecastError(fc, g.get("quantiles"));
                    if (!(fe <= TOLERANCE)) {
                        failures.add(testCase + " forecast: " + fe);
                    }
                    System.out.printf("%s: %d calls, worst step %.2e, forecast %.2e%n", testCase, windows.size(), worstOut, fe);
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    @Test
    public void batchingDoesNotChangeForecasts() throws IOException {
        try (T0Weights weights = T0Weights.load(T0Golden.checkpoint())) {
            T0Config c = weights.config();
            CpuT0 cpu = new CpuT0(weights);
            for (String testCase : List.of("ragged_batch", "covariates_batch")) {
                Map<String, Object> info = T0Golden.caseInfo(testCase);
                int h = ((Number) info.get("H")).intValue();
                try (SafeTensors g = T0Golden.open(testCase)) {
                    List<TimeSeries> series = T0Golden.series(g, info);
                    List<Forecast> one = new T0Pipeline(c, cpu, T0Golden.options(info), 1, Long.MAX_VALUE).forecast(series, h);
                    double fe = forecastError(one, g.get("quantiles"));
                    System.out.printf("%s one series per pass: forecast %.2e%n", testCase, fe);
                    assertTrue(testCase + " one series per pass: " + fe, fe <= TOLERANCE);
                }
            }
        }
    }

    /** null when the windows are identical (values within {@code tolerance}), else what differs. */
    static String sameWindow(T0Batch a, T0Batch e, double tolerance) {
        if (a.rows() != e.rows() || a.width() != e.width()) {
            return "shape " + a.rows() + "x" + a.width() + " != " + e.rows() + "x" + e.width();
        }
        if (!Arrays.equals(a.mask(), e.mask())) {
            return "masks differ";
        }
        if (!Arrays.equals(a.group(), e.group())) {
            return "groups differ";
        }
        if (!Arrays.equals(a.type(), e.type())) {
            return "types differ";
        }
        double v = T0Golden.relativeMaxError(a.values(), e.values(), null);
        return v <= tolerance ? null : "values " + v;
    }

    /** The golden {@code (targets, H, Q)} quantiles against the forecasts, every target of every series in order. */
    static double forecastError(List<Forecast> fc, Tensor golden) {
        int b = golden.dim(0), h = golden.dim(1), q = golden.dim(2);
        float[] exp = golden.toFloatArray(), act = new float[exp.length];
        int r = 0;
        for (Forecast f : fc) {
            for (int t = 0; t < f.targets(); t++, r++) {
                if (r >= b || f.horizon() != h || f.levels().length != q) {
                    return Double.POSITIVE_INFINITY;
                }
                for (int s = 0; s < h; s++) {
                    for (int k = 0; k < q; k++) {
                        act[(r * h + s) * q + k] = f.values()[t][k][s];
                    }
                }
            }
        }
        return r == b ? T0Golden.relativeMaxError(act, exp, null) : Double.POSITIVE_INFINITY;
    }
}
