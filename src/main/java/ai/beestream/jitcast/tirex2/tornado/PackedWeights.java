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
package ai.beestream.jitcast.tirex2.tornado;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.tirex2.TiRex2Weights;

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
 *   <li>the four headwise sLSTM gate projections (slot order {@code (fgate, igate, zgate, ogate)})
 *       → one block-diagonal {@code [4D, D]} matrix, so they run as a single GEMM whose output is
 *       laid out {@code [token][gate][head][Dh]};
 *   <li>variate attention {@code WQ | WK | WV} → one {@code [3D, D]} matrix.
 * </ul>
 */
public final class PackedWeights {

    /** A linear layer: weight at {@code w}, bias at {@code b} (or -1), {@code [out, in]}. */
    public record Lin(int w, int b, int in, int out) {}

    public record Residual(Lin hidden, Lin output, Lin residual) {}

    /** Offsets of one block; the unused cell's fields are -1 / null. */
    public record Block(boolean mlstm, int normLstm, Lin mlstmProj, int headNorm, Lin outProj, Lin slstmGates,
            int recurrent, int slstmBias, int groupNorm, Lin combination, int normFfn, Lin upGateUp, Lin down,
            int ffnDim, Variate variate, int recurrentSize) {}

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
            Lin gates = null;
            int headNorm = -1, rec = -1, bias = -1, groupNorm = -1;
            boolean mlstm = t.cell() instanceof TiRex2Weights.Mlstm;
            if (t.cell() instanceof TiRex2Weights.Mlstm m) {
                List<TiRex2Weights.Linear> parts = List.of(m.q(), m.k(), m.v(), m.o(), m.igate(), m.fgate());
                proj = p.fused(parts);
                headNorm = p.put(m.headNorm());
                outProj = p.linear(m.outProj());
            } else if (t.cell() instanceof TiRex2Weights.Slstm s) {
                gates = p.denseGates(s.gates());
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
            int recSize = t.cell() instanceof TiRex2Weights.Slstm sl ? Math.toIntExact(sl.recurrent().elements()) : 0;
            blocks.add(new Block(mlstm, normLstm, proj, headNorm, outProj, gates, rec, bias, groupNorm, combination,
                    normFfn, upGateUp, down, t.ffnUp().out(), variate, recSize));
        }
        int outNorm = p.put(w.stackOutNorm());
        Residual out = p.residual(w.outputEmbedding());
        return new PackedWeights(p.build(), in, List.copyOf(blocks), outNorm, out);
    }

    public FloatArray data() {
        return data;
    }

    private final java.util.Map<Lin, FloatArray> matrices = new java.util.HashMap<>();
    private final java.util.Map<Lin, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray> halfMatrices =
            new java.util.HashMap<>();

    private uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray recurrentHalf;
    private final java.util.Map<Block, Integer> recurrentHalfOffsets = new java.util.IdentityHashMap<>();

    /**
     * The sLSTM recurrent kernels R of every block, converted to FP16 and packed back to back
     * ({@link #recurrentHalfOffset} gives each block's start); built on first use.
     */
    public synchronized uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray recurrentHalf() {
        if (recurrentHalf == null) {
            int each = 0, count = 0;
            for (Block b : blocks) {
                if (!b.mlstm()) {
                    each = b.recurrentSize();
                    count++;
                }
            }
            recurrentHalf = new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(Math.max(1, each * count));
            MemorySegment src = data.getSegment(), dst = recurrentHalf.getSegment();
            int at = 0;
            for (Block b : blocks) {
                if (b.mlstm()) {
                    continue;
                }
                recurrentHalfOffsets.put(b, at);
                for (int i = 0; i < b.recurrentSize(); i++) {
                    float v = src.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, (long) b.recurrent() + i);
                    dst.setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, at + i, Float.floatToFloat16(v));
                }
                at += b.recurrentSize();
            }
        }
        return recurrentHalf;
    }

    /** Where block {@code b}'s R starts in {@link #recurrentHalf()}. */
    public int recurrentHalfOffset(Block b) {
        recurrentHalf();
        return recurrentHalfOffsets.get(b);
    }

    /** The weight of {@code l} as its own FP16 {@code [out, in]} array (converted once, then cached). */
    public synchronized uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray halfMatrix(Lin l) {
        return halfMatrices.computeIfAbsent(l, k -> {
            int n = k.out() * k.in();
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray m = new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(n);
            MemorySegment src = data.getSegment();
            MemorySegment dst = m.getSegment();
            for (int i = 0; i < n; i++) {
                float v = src.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, (long) k.w() + i);
                dst.setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, i, Float.floatToFloat16(v));
            }
            return m;
        });
    }

    /**
     * The weight of {@code l} as its own {@code [out, in]} array (copied once, then cached), for
     * native GEMMs such as cuBLAS that take whole arrays rather than an offset into the pack.
     */
    public synchronized FloatArray matrix(Lin l) {
        return matrices.computeIfAbsent(l, k -> {
            FloatArray m = new FloatArray(k.out() * k.in());
            MemorySegment.copy(data.getSegment(), 4L * k.w(), m.getSegment(), 0, 4L * k.out() * k.in());
            return m;
        });
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

        /**
         * {@code W[g·D + h·Dh + o][h·Dh + j] = gates[g][h][o][j]}, zero off the head blocks: the
         * headwise projections as one dense linear {@code [4D, D]} (4× the arithmetic, but one
         * GEMM instead of 16 small ones).
         */
        Lin denseGates(Tensor[] gates) {
            int heads = gates[0].dim(0), dh = gates[0].dim(1), d = heads * dh;
            float[] w = new float[4 * d * d];
            for (int g = 0; g < 4; g++) {
                float[] src = gates[g].toFloatArray();
                for (int h = 0; h < heads; h++) {
                    for (int o = 0; o < dh; o++) {
                        System.arraycopy(src, (h * dh + o) * dh, w, (g * d + h * dh + o) * d + h * dh, dh);
                    }
                }
            }
            return new Lin(put(w), -1, d, 4 * d);
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
