package ai.beestream.jitllm.tirex.tornado;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitllm.tirex.io.Tensor;
import ai.beestream.jitllm.tirex.model.TiRex2Weights;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Every TiRex-2 weight in one {@link FloatArray}, so a plan transfers a single buffer once and the
 * kernels address tensors by element offset. Projections that share an input are fused on the
 * way in:
 *
 * <ul>
 *   <li>mLSTM {@code q | k | v | o | i | f} → one {@code [1544, 512]} matrix, bias zero except
 *       for i and f;
 *   <li>SwiGLU {@code up_gate | up} → one {@code [2·ffn, 512]} matrix;
 *   <li>sLSTM gate projections in slot order {@code (fgate, igate, zgate, ogate)} → contiguous;
 *   <li>variate attention {@code WQ | WK | WV} → one {@code [3D, D]} matrix.
 * </ul>
 */
public final class PackedWeights {

    /** A linear layer: weight at {@code w}, bias at {@code b} (or -1), {@code [out, in]}. */
    public record Lin(int w, int b, int in, int out) {}

    public record Residual(Lin hidden, Lin output, Lin residual) {}

    /** Offsets of one block; the unused cell's fields are -1 / null. */
    public record Block(boolean mlstm, int normLstm, Lin mlstmProj, int headNorm, Lin outProj, int slstmGates,
            int recurrent, int slstmBias, int groupNorm, Lin combination, int normFfn, Lin upGateUp, Lin down,
            int ffnDim, Variate variate) {}

    /** The variate mixer: {@code WQ | WK | WV} fused into {@code qkv [3D, D]}, then WO and the MLP. */
    public record Variate(int normAttn, Lin qkv, Lin wo, int normFfn, Lin ffnIn, Lin ffnOut) {}

    private final FloatArray data;
    private final Residual inputEmbedding;
    private final List<Block> blocks;
    private final int stackOutNorm;
    private final Residual outputEmbedding;

    private PackedWeights(FloatArray data, Residual in, List<Block> blocks, int outNorm, Residual out) {
        this.data = data;
        this.inputEmbedding = in;
        this.blocks = blocks;
        this.stackOutNorm = outNorm;
        this.outputEmbedding = out;
    }

    public static PackedWeights pack(TiRex2Weights w) {
        Packer p = new Packer();
        Residual in = p.residual(w.inputEmbedding());
        List<Block> blocks = new ArrayList<>();
        for (TiRex2Weights.Block b : w.blocks()) {
            TiRex2Weights.TimeMixer t = b.time();
            int normLstm = p.put(t.normLstm());
            Lin proj = null, outProj = null;
            int headNorm = -1, gates = -1, rec = -1, bias = -1, groupNorm = -1;
            boolean mlstm = t.cell() instanceof TiRex2Weights.Mlstm;
            if (t.cell() instanceof TiRex2Weights.Mlstm m) {
                List<TiRex2Weights.Linear> parts = List.of(m.q(), m.k(), m.v(), m.o(), m.igate(), m.fgate());
                proj = p.fused(parts);
                headNorm = p.put(m.headNorm());
                outProj = p.linear(m.outProj());
            } else if (t.cell() instanceof TiRex2Weights.Slstm s) {
                gates = p.put(s.gates()[0]);
                for (int g = 1; g < 4; g++) {
                    p.put(s.gates()[g]);
                }
                rec = p.put(s.recurrent());
                bias = p.put(s.bias());
                groupNorm = p.put(s.groupNorm());
            }
            Lin combination = p.linear(t.combination());
            int normFfn = p.put(t.normFfn());
            Lin upGateUp = p.fused(List.of(t.ffnUpGate(), t.ffnUp()));
            Lin down = p.linear(t.ffnDown());
            TiRex2Weights.VariateMixer v = b.variate();
            Variate variate = new Variate(p.put(v.normAttn()), p.fused(List.of(v.wq(), v.wk(), v.wv())),
                    p.linear(v.wo()), p.put(v.normFfn()), p.linear(v.ffnIn()), p.linear(v.ffnOut()));
            blocks.add(new Block(mlstm, normLstm, proj, headNorm, outProj, gates, rec, bias, groupNorm, combination,
                    normFfn, upGateUp, down, t.ffnUp().out(), variate));
        }
        int outNorm = p.put(w.stackOutNorm());
        Residual out = p.residual(w.outputEmbedding());
        return new PackedWeights(p.build(), in, List.copyOf(blocks), outNorm, out);
    }

    public FloatArray data() {
        return data;
    }

    public Residual inputEmbedding() {
        return inputEmbedding;
    }

    public List<Block> blocks() {
        return blocks;
    }

    public int stackOutNorm() {
        return stackOutNorm;
    }

    public Residual outputEmbedding() {
        return outputEmbedding;
    }

    /** Lays tensors out one after another, then copies them into a FloatArray in one pass. */
    private static final class Packer {
        private final List<MemorySegment> pieces = new ArrayList<>();
        private final List<float[]> synthetic = new ArrayList<>();
        private final List<Integer> order = new ArrayList<>(); // >= 0: pieces index, < 0: ~synthetic index
        private long size;

        int put(Tensor t) {
            int at = offset();
            pieces.add(t.data());
            order.add(pieces.size() - 1);
            size += t.elements();
            return at;
        }

        int put(float[] v) {
            int at = offset();
            synthetic.add(v);
            order.add(~(synthetic.size() - 1));
            size += v.length;
            return at;
        }

        private int offset() {
            return Math.toIntExact(size);
        }

        Lin linear(TiRex2Weights.Linear l) {
            int w = put(l.weight());
            int b = l.bias() == null ? -1 : put(l.bias());
            return new Lin(w, b, l.in(), l.out());
        }

        Residual residual(TiRex2Weights.ResidualBlock r) {
            return new Residual(linear(r.hidden()), linear(r.output()), linear(r.residual()));
        }

        /** Stacks linears with the same input along the output dimension; missing biases are zeros. */
        Lin fused(List<TiRex2Weights.Linear> parts) {
            int in = parts.get(0).in();
            int out = 0;
            boolean anyBias = false;
            int w = -1;
            for (TiRex2Weights.Linear l : parts) {
                if (l.in() != in) {
                    throw new IllegalArgumentException("fused linears must share the input size");
                }
                int at = put(l.weight());
                w = w < 0 ? at : w;
                out += l.out();
                anyBias |= l.bias() != null;
            }
            int b = -1;
            if (anyBias) {
                float[] bias = new float[out];
                int o = 0;
                for (TiRex2Weights.Linear l : parts) {
                    if (l.bias() != null) {
                        float[] v = l.bias().toFloatArray();
                        System.arraycopy(v, 0, bias, o, v.length);
                    }
                    o += l.out();
                }
                b = put(bias);
            }
            return new Lin(w, b, in, out);
        }

        FloatArray build() {
            FloatArray a = new FloatArray(Math.toIntExact(size));
            MemorySegment dst = a.getSegment();
            long at = 0;
            for (int idx : order) {
                if (idx >= 0) {
                    MemorySegment src = pieces.get(idx);
                    MemorySegment.copy(src, 0, dst, at, src.byteSize());
                    at += src.byteSize();
                } else {
                    float[] v = synthetic.get(~idx);
                    MemorySegment.copy(MemorySegment.ofArray(v), 0, dst, at, 4L * v.length);
                    at += 4L * v.length;
                }
            }
            return a;
        }
    }
}
