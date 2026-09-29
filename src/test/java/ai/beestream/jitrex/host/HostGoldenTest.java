package ai.beestream.jitrex.host;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitrex.GoldenData;
import ai.beestream.jitrex.api.Forecast;
import ai.beestream.jitrex.api.TimeSeries;
import ai.beestream.jitrex.cpu.CpuTiRex2Engine;
import ai.beestream.jitrex.io.SafeTensors;
import ai.beestream.jitrex.model.ModelBatch;
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
                        List<TimeSeries> series = pass == 0 ? in : in.stream().map(ForecastPipeline::negate).toList();
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
