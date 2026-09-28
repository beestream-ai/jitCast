package ai.beestream.jitllm.tirex.cpu;

import ai.beestream.jitllm.tirex.model.TiRex2Config;
import ai.beestream.jitllm.tirex.model.TiRex2Weights;

/**
 * The FP32 CPU reference of {@code TiRex2.forward()}: the oracle the TornadoVM kernels are tested
 * against, stage by stage, with the same names as the golden activations.
 *
 * <p>P1 status: the input stage and the patch embedding are implemented; the block stack and the
 * output stage come next.
 */
public final class CpuReference {

    private final TiRex2Config config;
    private final CpuWeights.ResidualBlock inputEmbedding;

    public CpuReference(TiRex2Weights weights) {
        this.config = weights.config();
        this.inputEmbedding = CpuWeights.ResidualBlock.of(weights.inputEmbedding());
    }

    public TiRex2Config config() {
        return config;
    }

    /** {@code input_patch_embedding(cat(tokens, mask))}: {@code [rows · patches · D]}. */
    public float[] embed(InputStage.Result input) {
        return Ops.residualBlock(input.tokens(), input.rows() * input.patches(), inputEmbedding);
    }
}
