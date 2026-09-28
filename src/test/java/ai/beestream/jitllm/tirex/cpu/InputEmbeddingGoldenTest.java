package ai.beestream.jitllm.tirex.cpu;

import static org.junit.Assert.assertTrue;

import java.io.IOException;

import ai.beestream.jitllm.tirex.GoldenData;
import ai.beestream.jitllm.tirex.io.SafeTensors;
import ai.beestream.jitllm.tirex.model.TiRex2Weights;
import org.junit.Test;

/** Scaler + patching + patch embedding against {@code input_patch_embedding} in every golden case. */
public class InputEmbeddingGoldenTest {

    private static final double TOLERANCE = 1e-4;

    @Test
    public void matchesReferenceInEveryCase() throws IOException {
        try (TiRex2Weights weights = TiRex2Weights.load(GoldenData.weights())) {
            CpuReference ref = new CpuReference(weights);
            for (String testCase : GoldenData.cases()) {
                try (SafeTensors g = GoldenData.open(testCase)) {
                    CpuReference.Batch batch = GoldenData.batch(g, 0);
                    int rows = batch.x().length;
                    float[] actual = ref.embed(InputStage.apply(batch.x(), ref.config()));
                    float[] expected = g.get("act.pass0.input_patch_embedding#0.out").toFloatArray();
                    double err = GoldenData.relativeMaxError(actual, expected);
                    System.out.printf("%-26s rows=%3d  rel max err %.2e%n", testCase, rows, err);
                    assertTrue(testCase + ": relative error " + err, err < TOLERANCE);
                }
            }
        }
    }
}
