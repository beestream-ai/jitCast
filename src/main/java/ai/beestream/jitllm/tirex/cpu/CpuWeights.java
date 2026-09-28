package ai.beestream.jitllm.tirex.cpu;

import java.util.List;

import ai.beestream.jitllm.tirex.io.Tensor;
import ai.beestream.jitllm.tirex.model.TiRex2Weights;

/** Heap copies of the weights the CPU reference reads, taken once from the mapped file. */
final class CpuWeights {

    /** A linear layer: {@code w [out · in]}, {@code b [out]} or null. */
    record Linear(float[] w, float[] b, int in, int out) {
        static Linear of(TiRex2Weights.Linear l) {
            return new Linear(f(l.weight()), f(l.bias()), l.in(), l.out());
        }

        float[] apply(float[] x, int rows) {
            return Ops.linear(x, rows, w, b, in, out);
        }
    }

    record ResidualBlock(Linear hidden, Linear output, Linear residual) {
        static ResidualBlock of(TiRex2Weights.ResidualBlock w) {
            return new ResidualBlock(Linear.of(w.hidden()), Linear.of(w.output()), Linear.of(w.residual()));
        }
    }

    sealed interface Cell permits Mlstm, Slstm {}

    record Mlstm(Linear q, Linear k, Linear v, Linear o, Linear igate, Linear fgate, float[] headNorm, Linear outProj)
            implements Cell {}

    /** {@code gates[slot] [H·Dh(out)·Dh(in)]}, {@code r [H·Dh(prev)·4·Dh]}, {@code b [H·4·Dh]}. */
    record Slstm(float[][] gates, float[] r, float[] b, float[] groupNorm) implements Cell {}

    record Block(Cell cell, float[] normLstm, Linear combination, float[] normFfn, Linear upGate, Linear up,
            Linear down, float[] normAttn, Linear wq, Linear wk, Linear wv, Linear wo, float[] normFfnAttn,
            Linear ffnIn, Linear ffnOut) {

        static Block of(TiRex2Weights.Block b) {
            TiRex2Weights.TimeMixer t = b.time();
            TiRex2Weights.VariateMixer v = b.variate();
            Cell cell = switch (t.cell()) {
                case TiRex2Weights.Mlstm m -> new Mlstm(Linear.of(m.q()), Linear.of(m.k()), Linear.of(m.v()),
                        Linear.of(m.o()), Linear.of(m.igate()), Linear.of(m.fgate()), f(m.headNorm()),
                        Linear.of(m.outProj()));
                case TiRex2Weights.Slstm s -> {
                    float[][] gates = new float[4][];
                    for (int g = 0; g < 4; g++) {
                        gates[g] = f(s.gates()[g]);
                    }
                    yield new Slstm(gates, f(s.recurrent()), f(s.bias()), f(s.groupNorm()));
                }
            };
            return new Block(cell, f(t.normLstm()), Linear.of(t.combination()), f(t.normFfn()),
                    Linear.of(t.ffnUpGate()), Linear.of(t.ffnUp()), Linear.of(t.ffnDown()),
                    f(v.normAttn()), Linear.of(v.wq()), Linear.of(v.wk()), Linear.of(v.wv()), Linear.of(v.wo()),
                    f(v.normFfn()), Linear.of(v.ffnIn()), Linear.of(v.ffnOut()));
        }
    }

    final ResidualBlock inputEmbedding;
    final List<Block> blocks;
    final float[] stackOutNorm;
    final ResidualBlock outputEmbedding;

    CpuWeights(TiRex2Weights w) {
        inputEmbedding = ResidualBlock.of(w.inputEmbedding());
        blocks = w.blocks().stream().map(Block::of).toList();
        stackOutNorm = f(w.stackOutNorm());
        outputEmbedding = ResidualBlock.of(w.outputEmbedding());
    }

    static float[] f(Tensor t) {
        return t == null ? null : t.toFloatArray();
    }
}
