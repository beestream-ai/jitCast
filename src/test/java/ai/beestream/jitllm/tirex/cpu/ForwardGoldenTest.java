package ai.beestream.jitllm.tirex.cpu;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitllm.tirex.GoldenData;
import ai.beestream.jitllm.tirex.io.SafeTensors;
import ai.beestream.jitllm.tirex.model.TiRex2Weights;
import org.junit.Test;

/**
 * The full CPU forward pass against the reference, stage by stage (patch embedding, every time and
 * variate mixer, output norm and embedding) and on the model output, in every golden case.
 */
public class ForwardGoldenTest {

    private static final double TOLERANCE = 1e-4;

    @Test
    public void everyStageMatchesTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (TiRex2Weights weights = TiRex2Weights.load(GoldenData.weights())) {
            CpuReference ref = new CpuReference(weights);
            int seq = ref.config().numPatches(), d = ref.config().embeddingDim();
            for (String testCase : GoldenData.cases()) {
                try (SafeTensors g = GoldenData.open(testCase)) {
                    for (int pass = 0; pass < GoldenData.passes(testCase); pass++) {
                        CpuReference.Batch batch = GoldenData.batch(g, pass);
                        int rows = batch.x().length;
                        Map<String, float[]> stages = new LinkedHashMap<>();
                        long t0 = System.nanoTime();
                        float[][][] out = ref.forward(batch, stages::put);
                        long ms = (System.nanoTime() - t0) / 1_000_000;

                        double worst = 0;
                        String worstStage = "";
                        String prefix = "act.pass" + pass + ".";
                        for (Map.Entry<String, float[]> e : stages.entrySet()) {
                            float[] expected = g.get(prefix + e.getKey() + "#0.out").toFloatArray();
                            float[] actual = e.getKey().endsWith("variate_mixer")
                                    ? GoldenData.swap01(e.getValue(), rows, seq, d)
                                    : e.getValue();
                            double err = GoldenData.relativeMaxError(actual, expected);
                            if (err > worst) {
                                worst = err;
                                worstStage = e.getKey();
                            }
                            if (err > TOLERANCE) {
                                failures.add(testCase + " pass" + pass + " " + e.getKey() + ": " + err);
                            }
                        }
                        float[] flat = new float[rows * out[0].length * out[0][0].length];
                        int i = 0;
                        for (float[][] row : out) {
                            for (float[] q : row) {
                                System.arraycopy(q, 0, flat, i, q.length);
                                i += q.length;
                            }
                        }
                        double outErr = GoldenData.relativeMaxError(flat, g.get("pass" + pass + ".model.out").toFloatArray());
                        if (outErr > TOLERANCE) {
                            failures.add(testCase + " pass" + pass + " model.out: " + outErr);
                        }
                        System.out.printf("%-26s pass%d rows=%2d %6d ms  model.out %.2e  worst stage %s %.2e%n",
                                testCase, pass, rows, ms, outErr, worstStage, worst);
                    }
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }
}
