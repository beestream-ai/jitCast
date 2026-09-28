package ai.beestream.jitllm.tirex.cpu;

import ai.beestream.jitllm.tirex.io.Tensor;
import ai.beestream.jitllm.tirex.model.TiRex2Weights;

/** Heap copies of the weights the CPU reference reads, taken once from the mapped file. */
final class CpuWeights {

    record ResidualBlock(float[] hiddenW, float[] hiddenB, float[] outputW, float[] outputB,
            float[] residualW, float[] residualB, int in, int hidden, int out) {

        static ResidualBlock of(TiRex2Weights.ResidualBlock w) {
            return new ResidualBlock(
                    f(w.hidden().weight()), f(w.hidden().bias()),
                    f(w.output().weight()), f(w.output().bias()),
                    f(w.residual().weight()), f(w.residual().bias()),
                    w.hidden().in(), w.hidden().out(), w.output().out());
        }
    }

    private CpuWeights() {}

    static float[] f(Tensor t) {
        return t == null ? null : t.toFloatArray();
    }
}
