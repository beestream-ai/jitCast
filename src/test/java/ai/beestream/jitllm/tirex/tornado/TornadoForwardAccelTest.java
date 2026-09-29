package ai.beestream.jitllm.tirex.tornado;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitllm.tirex.GoldenData;
import ai.beestream.jitllm.tirex.api.Forecast;
import ai.beestream.jitllm.tirex.api.TimeSeries;
import ai.beestream.jitllm.tirex.io.SafeTensors;
import ai.beestream.jitllm.tirex.model.ModelBatch;
import org.junit.Test;

/**
 * The TornadoVM forward pass and engine against the reference, on every golden case (univariate,
 * multi-target, past and future covariates, mixed batches). Runs under -Paccel-tests; pick the device with
 * {@code -Dtirex.tornado.device=<driver>:<device>}.
 */
public class TornadoForwardAccelTest {

    private static final double TOLERANCE = 1e-4;

    @Test
    public void forwardAndForecastMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        int checked = 0;
        try (TornadoTiRex2Engine engine = TornadoTiRex2Engine.load(GoldenData.weights())) {
            for (String testCase : GoldenData.cases()) {
                try (SafeTensors g = GoldenData.open(testCase)) {
                    ModelBatch batch = GoldenData.batch(g, 0);
                    checked++;
                    long t0 = System.nanoTime();
                    float[][][] out = engine.forwardPass().forward(batch);
                    long first = (System.nanoTime() - t0) / 1_000_000;
                    t0 = System.nanoTime();
                    out = engine.forwardPass().forward(batch);
                    double again = (System.nanoTime() - t0) / 1e6;
                    double err = GoldenData.relativeMaxError(flatten(out), g.get("pass0.model.out").toFloatArray());

                    Map<String, Object> info = GoldenData.caseInfo(testCase);
                    int h = ((Number) info.get("H")).intValue();
                    @SuppressWarnings("unchecked")
                    Map<String, Object> kw = (Map<String, Object>) info.get("kwargs");
                    boolean signFlip = kw.containsKey("tta_sign_flip") ? Boolean.TRUE.equals(kw.get("tta_sign_flip"))
                            : engine.pipeline().signFlip();
                    List<TimeSeries> in = GoldenData.inputs(g);
                    List<Forecast> f = engine.pipeline().withSignFlip(signFlip).forecast(in, h);
                    double ferr = 0;
                    for (int s = 0; s < in.size(); s++) {
                        ferr = Math.max(ferr, GoldenData.relativeMaxError(flatten(f.get(s).values()),
                                g.get("forecast." + s).toFloatArray()));
                    }
                    System.out.printf("%-26s rows=%2d  first %6d ms  again %8.2f ms  model.out %.2e  forecast %.2e%n",
                            testCase, batch.rows(), first, again, err, ferr);
                    if (err > TOLERANCE || ferr > TOLERANCE) {
                        failures.add(testCase + ": model.out " + err + ", forecast " + ferr);
                    }
                }
            }
        }
        assertTrue("no case in scope", checked > 0);
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    static float[] flatten(float[][][] v) {
        float[] flat = new float[v.length * v[0].length * v[0][0].length];
        int i = 0;
        for (float[][] a : v) {
            for (float[] b : a) {
                System.arraycopy(b, 0, flat, i, b.length);
                i += b.length;
            }
        }
        return flat;
    }
}
