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
package ai.beestream.jitrex.tirex2;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitrex.api.Forecast;
import ai.beestream.jitrex.api.TimeSeries;
import ai.beestream.jitrex.host.SignFlip;
import ai.beestream.jitrex.io.SafeTensors;
import ai.beestream.jitrex.tirex2.cpu.CpuTiRex2Engine;
import org.junit.Test;

/**
 * The host side against the reference: the assembled batch must equal what TiRex2.forward()
 * received bit for bit (so trend detection, differencing, grouping and padding agree), and the
 * CPU engine's forecasts must match {@code forecast.<i>} in every golden case.
 */
public class HostGoldenTest {

    private static final double FORECAST_TOLERANCE = 1e-4;

    @Test
    public void assembledBatchEqualsTheReferenceBatch() throws IOException {
        try (CpuTiRex2Engine engine = CpuTiRex2Engine.load(GoldenData.weights())) {
            for (String testCase : GoldenData.cases()) {
                Map<String, Object> info = GoldenData.caseInfo(testCase);
                int h = ((Number) info.get("H")).intValue();
                try (SafeTensors g = GoldenData.open(testCase)) {
                    List<TimeSeries> in = GoldenData.inputs(g);
                    for (int pass = 0; pass < GoldenData.passes(testCase); pass++) {
                        List<TimeSeries> series = pass == 0 ? in : in.stream().map(SignFlip::negate).toList();
                        ForecastPipeline.Assembled assembled = engine.pipeline().assemble(series, Math.min(h, 320));
                        ModelBatch actual = assembled.batch();
                        if (testCase.equals("trend_diff") && pass == 0 && engine.pipeline().trendDiff()) {
                            assertTrue("trend_diff must take the differencing path", assembled.differenced()[0][0]);
                        }
                        ModelBatch expected = GoldenData.batch(g, pass);
                        assertEquals(testCase + " rows", expected.rows(), actual.rows());
                        for (int r = 0; r < expected.rows(); r++) {
                            // assertArrayEquals on float[] treats NaN == NaN
                            assertArrayEquals(testCase + " pass" + pass + " row " + r, expected.x()[r], actual.x()[r], 0f);
                        }
                        assertArrayEquals(testCase + " group", expected.group(), actual.group(), 0f);
                        for (int r = 0; r < expected.rows(); r++) {
                            assertEquals(testCase + " target " + r, expected.target()[r], actual.target()[r]);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void forecastsDoNotDependOnHowABatchIsSplit() throws IOException {
        try (CpuTiRex2Engine engine = CpuTiRex2Engine.load(GoldenData.weights())) {
            int checked = 0;
            for (String testCase : GoldenData.cases()) {
                int h = ((Number) GoldenData.caseInfo(testCase).get("H")).intValue();
                try (SafeTensors g = GoldenData.open(testCase)) {
                    List<TimeSeries> in = GoldenData.inputs(g);
                    if (in.size() < 2) {
                        continue;
                    }
                    List<Forecast> whole = engine.pipeline().forecast(in, h);
                    // a budget of one token runs every series in a forward pass of its own
                    List<Forecast> split = engine.pipeline().withMaxTokens(1).forecast(in, h);
                    for (int s = 0; s < in.size(); s++) {
                        float[][][] a = whole.get(s).values(), b = split.get(s).values();
                        for (int t = 0; t < a.length; t++) {
                            for (int q = 0; q < a[t].length; q++) {
                                assertArrayEquals(testCase + " series " + s, a[t][q], b[t][q], 0f);
                            }
                        }
                    }
                    checked++;
                }
            }
            assertTrue("no multi-series golden case", checked > 0);
        }
    }

    @Test
    public void forecastsMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (CpuTiRex2Engine engine = CpuTiRex2Engine.load(GoldenData.weights())) {
            for (String testCase : GoldenData.cases()) {
                Map<String, Object> info = GoldenData.caseInfo(testCase);
                int h = ((Number) info.get("H")).intValue();
                @SuppressWarnings("unchecked")
                Map<String, Object> kwargs = (Map<String, Object>) info.get("kwargs");
                // the case's own tta_sign_flip, else the checkpoint's default (as TiRex2.predict)
                boolean signFlip = kwargs.containsKey("tta_sign_flip") ? Boolean.TRUE.equals(kwargs.get("tta_sign_flip"))
                        : engine.pipeline().signFlip();
                ForecastPipeline pipeline = engine.pipeline().withSignFlip(signFlip);
                try (SafeTensors g = GoldenData.open(testCase)) {
                    List<TimeSeries> in = GoldenData.inputs(g);
                    List<Forecast> out = pipeline.forecast(in, h);
                    double worst = 0;
                    for (int s = 0; s < in.size(); s++) {
                        float[][][] v = out.get(s).values();
                        float[] flat = new float[v.length * v[0].length * v[0][0].length];
                        int i = 0;
                        for (float[][] target : v) {
                            for (float[] q : target) {
                                System.arraycopy(q, 0, flat, i, q.length);
                                i += q.length;
                            }
                        }
                        double err = GoldenData.relativeMaxError(flat, g.get("forecast." + s).toFloatArray());
                        worst = Math.max(worst, err);
                        if (err > FORECAST_TOLERANCE) {
                            failures.add(testCase + " series " + s + ": " + err);
                        }
                    }
                    System.out.printf("%-26s series=%d H=%3d  forecast rel max err %.2e%n", testCase, in.size(), h, worst);
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }
}
