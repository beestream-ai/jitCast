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
package ai.beestream.jitcast.timestral;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.timestral.cpu.CpuTimestral;
import org.junit.Test;

/**
 * The host side against {@code TimestralForecaster.forecast()}: every {@code predict_block} call
 * must receive what the reference's received (assembly, padding, the context cap, feedback and the
 * mirrored pass agree), and the CPU engine's forecasts must match the reference's in every golden
 * case. The context of a block is compared bit for bit; the medians fed back from earlier blocks
 * are the engine's own output and are compared within {@link #FEEDBACK_TOLERANCE}.
 */
public class PipelineGoldenTest {

    private static final double TOLERANCE = 1e-4;
    private static final double FEEDBACK_TOLERANCE = 1e-5;

    @Test
    public void callsAndForecastsMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TimestralWeights weights = TimestralWeights.load(TimestralGolden.checkpoint())) {
            CpuTimestral cpu = new CpuTimestral(weights);
            TimestralConfig c = weights.config();
            for (String testCase : TimestralGolden.cases()) {
                Map<String, Object> info = TimestralGolden.caseInfo(testCase);
                int h = ((Number) info.get("H")).intValue();
                try (SafeTensors g = TimestralGolden.open(testCase)) {
                    List<BlockInput> calls = new ArrayList<>();
                    TimestralPipeline pipeline = new TimestralPipeline(c, b -> {
                        calls.add(b);
                        return cpu.predictBlock(b);
                    }).withOptions(options(c, info));
                    List<Forecast> fc = pipeline.forecast(series(g, false), h);

                    int expectedCalls = TimestralGolden.calls(g);
                    if (calls.size() != expectedCalls) {
                        failures.add(testCase + ": " + calls.size() + " predict_block calls, expected " + expectedCalls);
                    }
                    int initial = calls.isEmpty() ? 0 : calls.get(0).length();
                    for (int k = 0; k < Math.min(calls.size(), expectedCalls); k++) {
                        String diff = compare(calls.get(k), TimestralGolden.block(g, k, c.patchSize()), initial);
                        if (diff != null) {
                            failures.add(testCase + " call" + k + ": " + diff);
                        }
                    }
                    double err = forecastError(fc, g.get("forecast"));
                    System.out.printf("%-24s H=%5d calls=%d forecast rel max err %.2e%n", testCase, h, calls.size(), err);
                    if (!(err <= TOLERANCE)) {
                        failures.add(testCase + " forecast: " + err);
                    }
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    /** A ragged batch passed as series of their own lengths forecasts as the NaN-padded batch does. */
    @Test
    public void raggedSeriesMatchTheirPaddedBatch() throws IOException {
        try (TimestralWeights weights = TimestralWeights.load(TimestralGolden.checkpoint());
                SafeTensors g = TimestralGolden.open("ragged_batch")) {
            Map<String, Object> info = TimestralGolden.caseInfo("ragged_batch");
            int h = ((Number) info.get("H")).intValue();
            TimestralPipeline pipeline = new TimestralPipeline(weights.config(), new CpuTimestral(weights));
            List<TimeSeries> ragged = series(g, true);
            assertTrue("the case is ragged", ragged.stream().mapToInt(TimeSeries::length).distinct().count() > 1);
            double err = forecastError(pipeline.forecast(ragged, h), g.get("forecast"));
            System.out.printf("ragged series            forecast rel max err %.2e%n", err);
            assertTrue("ragged forecast: " + err, err <= TOLERANCE);
        }
    }

    public static TimestralPipeline.Options options(TimestralConfig c, Map<String, Object> info) {
        TimestralPipeline.Options o = TimestralPipeline.Options.defaults(c);
        Object kw = info.get("kwargs");
        if (kw instanceof Map<?, ?> m && m.get("symmetric_averaging") instanceof Boolean b) {
            o = o.withSymmetricAveraging(b);
        }
        return o;
    }

    /** The golden inputs as series; with {@code stripLeadingNaN}, each series without its all-NaN prefix. */
    public static List<TimeSeries> series(SafeTensors g, boolean stripLeadingNaN) {
        float[][][] ctx = cube(g.get("in.context"));
        float[][][] past = g.contains("in.past_covariates") ? cube(g.get("in.past_covariates")) : null;
        float[][][] known = g.contains("in.known_covariates") ? cube(g.get("in.known_covariates")) : null;
        List<TimeSeries> out = new ArrayList<>();
        for (int b = 0; b < ctx.length; b++) {
            int start = 0;
            if (stripLeadingNaN) {
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

    /** Relative max error of the forecasts against the golden {@code (B, V, H, Q)} quantiles. */
    public static double forecastError(List<Forecast> fc, Tensor golden) {
        int bs = golden.dim(0), vs = golden.dim(1), hs = golden.dim(2), qs = golden.dim(3);
        float[] exp = golden.toFloatArray();
        float[] act = new float[exp.length];
        for (int b = 0; b < bs; b++) {
            for (int v = 0; v < vs; v++) {
                for (int t = 0; t < hs; t++) {
                    for (int q = 0; q < qs; q++) {
                        act[((b * vs + v) * hs + t) * qs + q] = fc.get(b).values()[v][q][t];
                    }
                }
            }
        }
        return TimestralGolden.relativeMaxError(act, exp);
    }

    /**
     * null if the call's arguments match the reference's, else what differs. A block whose context
     * is longer than the first block's carries fed-back medians in its last {@code length − initial}
     * steps (fewer once the context is capped); those are compared within {@link #FEEDBACK_TOLERANCE}
     * and every other value bit for bit.
     */
    private static String compare(BlockInput a, BlockInput e, int initial) {
        if (a.horizonPatches() != e.horizonPatches()) {
            return "horizon patches " + a.horizonPatches() + " != " + e.horizonPatches();
        }
        if (a.rows() != e.rows() || a.length() != e.length()) {
            return "shape " + a.rows() + "x" + a.length() + " != " + e.rows() + "x" + e.length();
        }
        int fed = Math.max(0, a.length() - initial);
        if (fed == 0 && !Arrays.deepEquals(a.inputs(), e.inputs())) {
            return "inputs differ" + firstDiff(a.inputs(), e.inputs());
        }
        if (fed > 0) {
            int ctx = a.length() - fed;
            float[][] ac = new float[a.rows()][], ec = new float[a.rows()][];
            float[] af = new float[a.rows() * fed], ef = new float[a.rows() * fed];
            for (int i = 0; i < a.rows(); i++) {
                ac[i] = Arrays.copyOf(a.inputs()[i], ctx);
                ec[i] = Arrays.copyOf(e.inputs()[i], ctx);
                System.arraycopy(a.inputs()[i], ctx, af, i * fed, fed);
                System.arraycopy(e.inputs()[i], ctx, ef, i * fed, fed);
            }
            if (!Arrays.deepEquals(ac, ec)) {
                return "context differs" + firstDiff(ac, ec);
            }
            double err = TimestralGolden.relativeMaxError(af, ef);
            if (!(err <= FEEDBACK_TOLERANCE)) {
                return "fed-back medians differ: " + err;
            }
        }
        if (!Arrays.deepEquals(a.observed(), e.observed())) {
            return "observation masks differ";
        }
        if (!Arrays.equals(a.group(), e.group())) {
            return "group ids differ";
        }
        if (!Arrays.equals(a.role(), e.role())) {
            return "roles differ: " + Arrays.toString(a.role()) + " vs " + Arrays.toString(e.role());
        }
        if (!Arrays.deepEquals(a.knownFuture(), e.knownFuture()) || !Arrays.deepEquals(a.knownFutureMask(), e.knownFutureMask())) {
            return "known futures differ";
        }
        return null;
    }

    private static String firstDiff(float[][] a, float[][] e) {
        if (a.length != e.length || a[0].length != e[0].length) {
            return " in shape: " + a.length + "x" + a[0].length + " vs " + e.length + "x" + e[0].length;
        }
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                if (Float.floatToIntBits(a[i][j]) != Float.floatToIntBits(e[i][j])) {
                    return " at [" + i + "][" + j + "]: " + a[i][j] + " vs " + e[i][j];
                }
            }
        }
        return "";
    }
}
