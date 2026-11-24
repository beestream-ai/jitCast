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
package ai.beestream.jitcast.t0.tornado;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.t0.T0Config;
import ai.beestream.jitcast.t0.T0Weights;
import ai.beestream.jitcast.t0.cpu.CpuT0;
import ai.beestream.jitcast.tornado.Kernels;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Every t0 weight in one device buffer, addressed by element offsets, with the rotary embedding's
 * frequencies and the logarithms of the xPos base scales precomputed. The library GEMMs (cuBLAS) cannot
 * address an offset into that buffer, so each matrix also has its own FP32 or FP16 copy, made on first use
 * ({@link #matrix}, {@link #halfMatrix}).
 */
public final class T0Packed {

    /**
     * A linear layer in the buffer.
     *
     * @param w the offset of {@code W [out, in]}
     * @param b the offset of the bias
     * @param in the input width
     * @param out the output width
     */
    public record Lin(int w, int b, int in, int out) {}

    /**
     * {@code output(ReLU(hidden(x))) + residual(x)}.
     *
     * @param hidden the first layer
     * @param output the second layer
     * @param residual the linear shortcut
     */
    public record Residual(Lin hidden, Lin output, Lin residual) {}

    /**
     * One transformer layer; norms are offsets into the buffer.
     *
     * @param group whether the layer attends across variates
     * @param attnNorm the RMSNorm scale before attention
     * @param qkv the fused projection
     * @param qNorm the per-head query RMSNorm scale
     * @param kNorm the per-head key RMSNorm scale
     * @param out the attention output projection
     * @param mlpNorm the RMSNorm scale before the feed-forward
     * @param up the feed-forward input layer (gate, then value)
     * @param down the feed-forward output layer
     */
    public record Layer(boolean group, int attnNorm, Lin qkv, int qNorm, int kNorm, Lin out, int mlpNorm, Lin up, Lin down) {}

    private final FloatArray data;
    private final Residual encoder;
    private final int typeEmbeddings;
    private final List<Layer> layers;
    private final int outNorm;
    private final Residual decoder;
    private final int invFreq;
    private final int logXpos;
    private final Map<Lin, FloatArray> matrices = new IdentityHashMap<>();
    private final Map<Lin, HalfFloatArray> halfMatrices = new IdentityHashMap<>();

    private T0Packed(FloatArray data, Residual encoder, int typeEmbeddings, List<Layer> layers, int outNorm, Residual decoder,
            int invFreq, int logXpos) {
        this.data = data;
        this.encoder = encoder;
        this.typeEmbeddings = typeEmbeddings;
        this.layers = layers;
        this.outNorm = outNorm;
        this.decoder = decoder;
        this.invFreq = invFreq;
        this.logXpos = logXpos;
    }

    /**
     * {@return the weights packed into one buffer}
     *
     * @param w the checkpoint
     */
    public static T0Packed pack(T0Weights w) {
        T0Config c = w.config();
        Packer p = new Packer();
        Residual encoder = p.residual(w.patchEncoder());
        int types = p.add(w.typeEmbeddings().toFloatArray());
        List<Layer> layers = new ArrayList<>();
        for (T0Weights.Layer l : w.layers()) {
            layers.add(new Layer(l.group(), p.add(l.attnNorm().toFloatArray()), p.lin(l.qkv()), p.add(l.queryNorm().toFloatArray()),
                    p.add(l.keyNorm().toFloatArray()), p.lin(l.out()), p.add(l.mlpNorm().toFloatArray()), p.lin(l.up()),
                    p.lin(l.down())));
        }
        int outNorm = p.add(w.outNorm().toFloatArray());
        Residual decoder = p.residual(w.decoder());
        int invFreq = p.add(CpuT0.inverseFrequencies(c.headDim()));
        float[] base = CpuT0.xposBase(c.headDim());
        float[] logs = new float[base.length];
        for (int i = 0; i < logs.length; i++) {
            logs[i] = (float) Math.log(base[i]);
        }
        int logXpos = p.add(logs);
        return new T0Packed(p.build(), encoder, types, List.copyOf(layers), outNorm, decoder, invFreq, logXpos);
    }

    /** {@return the device buffer} */
    public FloatArray data() {
        return data;
    }

    /** {@return the patch encoder} */
    public Residual encoder() {
        return encoder;
    }

    /** {@return the offset of the type embeddings, {@code [3, D]}} */
    public int typeEmbeddings() {
        return typeEmbeddings;
    }

    /** {@return the transformer layers} */
    public List<Layer> layers() {
        return layers;
    }

    /** {@return the offset of the final RMSNorm scale} */
    public int outNorm() {
        return outNorm;
    }

    /** {@return the decoder} */
    public Residual decoder() {
        return decoder;
    }

    /** {@return the offset of the rotary embedding's per-pair frequencies, {@code headDim / 2} of them} */
    public int invFreq() {
        return invFreq;
    }

    /** {@return the offset of the logarithms of the xPos per-pair base scales, {@code headDim / 2} of them} */
    public int logXpos() {
        return logXpos;
    }

    /**
     * {@return the weight matrix {@code [out, in]} of {@code l} in a buffer of its own, for library GEMMs}
     *
     * @param l the layer
     */
    public synchronized FloatArray matrix(Lin l) {
        return matrices.computeIfAbsent(l, x -> {
            FloatArray m = new FloatArray(x.out() * x.in());
            for (int i = 0; i < m.getSize(); i++) {
                m.set(i, data.get(x.w() + i));
            }
            return m;
        });
    }

    /**
     * {@return the weight matrix {@code [out, in]} of {@code l} in FP16, for the FP16 GEMMs}
     *
     * @param l the layer
     */
    public synchronized HalfFloatArray halfMatrix(Lin l) {
        return halfMatrices.computeIfAbsent(l, x -> {
            HalfFloatArray m = new HalfFloatArray(x.out() * x.in());
            for (int i = 0; i < m.getSize(); i++) {
                m.set(i, new HalfFloat(Math.clamp(data.get(x.w() + i), -Kernels.HALF_MAX, Kernels.HALF_MAX)));
            }
            return m;
        });
    }

    /** Appends arrays and records their offsets. */
    private static final class Packer {
        private final List<float[]> parts = new ArrayList<>();
        private long size;

        int add(float[] v) {
            if (size + v.length > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("the checkpoint is too large for one device buffer");
            }
            int off = (int) size;
            parts.add(v);
            size += v.length;
            return off;
        }

        Lin lin(T0Weights.Linear l) {
            int w = add(l.weight().toFloatArray());
            int b = add(l.bias().toFloatArray());
            return new Lin(w, b, l.in(), l.out());
        }

        Residual residual(T0Weights.ResidualBlock r) {
            return new Residual(lin(r.hidden()), lin(r.output()), lin(r.residual()));
        }

        FloatArray build() {
            FloatArray f = new FloatArray((int) size);
            int off = 0;
            for (float[] v : parts) {
                java.lang.foreign.MemorySegment.copy(v, 0, f.getSegment(), java.lang.foreign.ValueLayout.JAVA_FLOAT,
                        (long) off * Float.BYTES, v.length);
                off += v.length;
            }
            return f;
        }
    }
}
