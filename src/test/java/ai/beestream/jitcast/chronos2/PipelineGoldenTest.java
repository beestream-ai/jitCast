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

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.chronos2.cpu.CpuChronos2;
import ai.beestream.jitcast.io.SafeTensors;
import org.junit.Test;

/**
 * The host side against {@code Chronos2Pipeline.predict()}: every forward call must receive what
 * the reference's received (batching, padding, the context cap, groups, the future covariates and
 * the long-horizon unrolling agree), and the CPU engine's forecasts must match the reference's in
 * every golden case. A first pass's context is compared bit for bit; an unrolled pass's context
 * carries interpolated forecasts and is compared within {@link #FEEDBACK_TOLERANCE}.
 */
public class PipelineGoldenTest {

    private static final double TOLERANCE = 1e-4;
    private static final double FEEDBACK_TOLERANCE = 1e-5;

    @Test
    public void callsAndForecastsMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (Chronos2Weights weights = Chronos2Weights.load(Chronos2Golden.checkpoint())) {
            CpuChronos2 cpu = new CpuChronos2(weights);
            Chronos2Config c = weights.config();
            for (String testCase : Chronos2Golden.cases()) {
                Map<String, Object> info = Chronos2Golden.caseInfo(testCase);
                int h = Chronos2Golden.horizon(info);
                try (SafeTensors g = Chronos2Golden.open(testCase)) {
                    List<ModelInput> calls = new ArrayList<>();
                    Chronos2Pipeline pipeline = new Chronos2Pipeline(c, in -> {
                        calls.add(in);
                        return cpu.forward(in);
                    }).withOptions(Chronos2Golden.options(c, info));
                    List<Forecast> fc = pipeline.forecast(Chronos2Golden.series(g, info), h);

                    int expectedCalls = Chronos2Golden.calls(g);
                    if (calls.size() != expectedCalls) {
                        failures.add(testCase + ": " + calls.size() + " forward calls, expected " + expectedCalls);
                    }
                    for (int k = 0; k < Math.min(calls.size(), expectedCalls); k++) {
                        String diff = compare(calls.get(k), Chronos2Golden.input(g, k, c.patchSize()), k == 0);
                        if (diff != null) {
                            failures.add(testCase + " call" + k + ": " + diff);
                        }
                    }
                    double err = Chronos2Golden.forecastError(fc, g.get("forecast"));
                    System.out.printf("%-22s H=%5d calls=%d forecast rel max err %.2e%n", testCase, h, calls.size(), err);
                    if (!(err <= TOLERANCE)) {
                        failures.add(testCase + " forecast: " + err);
                    }
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    /** A pass over the token budget runs as several, split between groups, with the same forecasts. */
    @Test
    public void splitPassesMatchOnePass() throws IOException {
        try (Chronos2Weights weights = Chronos2Weights.load(Chronos2Golden.checkpoint());
                SafeTensors g = Chronos2Golden.open("multi_step")) {
            Map<String, Object> info = Chronos2Golden.caseInfo("multi_step");
            int h = Chronos2Golden.horizon(info);
            CpuChronos2 cpu = new CpuChronos2(weights);
            List<Integer> rows = new ArrayList<>();
            Chronos2Pipeline pipeline = new Chronos2Pipeline(weights.config(), in -> {
                rows.add(in.rows());
                return cpu.forward(in);
            }).withMaxTokens(400);
            List<TimeSeries> series = Chronos2Golden.series(g, info);
            double err = Chronos2Golden.forecastError(pipeline.forecast(series, h), g.get("forecast"));
            System.out.printf("split passes %s forecast rel max err %.2e%n", rows, err);
            assertTrue("the unrolled pass is split", rows.size() > 2);
            assertTrue("split forecast: " + err, err <= TOLERANCE);
        }
    }

    /** null if the call's arguments match the reference's, else what differs. */
    private static String compare(ModelInput a, ModelInput e, boolean exact) {
        if (a.outputPatches() != e.outputPatches()) {
            return "output patches " + a.outputPatches() + " != " + e.outputPatches();
        }
        if (a.rows() != e.rows() || a.length() != e.length()) {
            return "shape " + a.rows() + "x" + a.length() + " != " + e.rows() + "x" + e.length();
        }
        if (!Arrays.equals(a.group(), e.group())) {
            return "group ids differ";
        }
        if (exact) {
            if (!Arrays.deepEquals(a.context(), e.context())) {
                return "contexts differ";
            }
        } else {
            float[] af = new float[a.rows() * a.length()], ef = new float[af.length];
            for (int i = 0; i < a.rows(); i++) {
                System.arraycopy(a.context()[i], 0, af, i * a.length(), a.length());
                System.arraycopy(e.context()[i], 0, ef, i * a.length(), a.length());
            }
            double err = Chronos2Golden.relativeMaxError(af, ef);
            if (!(err <= FEEDBACK_TOLERANCE)) {
                return "contexts differ: " + err;
            }
        }
        if (!Arrays.deepEquals(a.futureCovariates(), e.futureCovariates())) {
            return "future covariates differ";
        }
        return null;
    }
}
