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
package ai.beestream.jitcast.chronos2.tornado;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.chronos2.Chronos2Config;
import ai.beestream.jitcast.chronos2.Chronos2Weights;
import ai.beestream.jitcast.tornado.Kernels;

import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Every Chronos-2 weight in one device buffer, addressed by element offsets, with each attention
 * sublayer's q, k and v projections fused into one {@code q|k|v} matrix
 * ({@link Chronos2Weights#qkv}). The library GEMMs (cuBLAS) cannot address an offset into that
 * buffer, so each matrix also has its own FP32 or FP16 copy, made on first use ({@link #matrix},
 * {@link #halfMatrix}).
 */
public final class Chronos2Packed {

    /** A linear layer in the buffer: {@code W [out, in]} at {@code w}, the bias at {@code b} or −1. */
    public record Lin(int w, int b, int in, int out) {}

    public record Residual(Lin hidden, Lin output, Lin residual) {}

    public record Attention(int norm, Lin qkv, Lin o) {}

    public record Layer(Attention time, Attention group, int mlpNorm, Lin wi, Lin wo) {}

    private final FloatArray data;
    private final int reg;
    private final Residual inputEmbedding;
    private final List<Layer> layers;
    private final int finalNorm;
    private final Residual outputEmbedding;
    private final Map<Lin, FloatArray> matrices = new IdentityHashMap<>();
    private final Map<Lin, HalfFloatArray> halfMatrices = new IdentityHashMap<>();

    private Chronos2Packed(FloatArray data, int reg, Residual inputEmbedding, List<Layer> layers, int finalNorm,
            Residual outputEmbedding) {
        this.data = data;
        this.reg = reg;
        this.inputEmbedding = inputEmbedding;
        this.layers = layers;
        this.finalNorm = finalNorm;
        this.outputEmbedding = outputEmbedding;
    }

    public static Chronos2Packed pack(Chronos2Weights w) {
        Chronos2Config c = w.config();
        Packer p = new Packer();
        int reg = c.useRegToken() ? p.add(w.regToken()) : -1;
        Residual in = p.residual(w.inputEmbedding());
        List<Layer> layers = new ArrayList<>();
        for (Chronos2Weights.Layer l : w.layers()) {
            Attention time = p.attention(w, l.time(), c);
            Attention group = p.attention(w, l.group(), c);
            int mlpNorm = p.add(l.mlpNorm().toFloatArray());
            layers.add(new Layer(time, group, mlpNorm, p.lin(l.wi()), p.lin(l.wo())));
        }
        int finalNorm = p.add(w.finalNorm().toFloatArray());
        Residual out = p.residual(w.outputEmbedding());
        return new Chronos2Packed(p.build(), reg, in, List.copyOf(layers), finalNorm, out);
    }

    public FloatArray data() {
        return data;
    }

    /** The offset of the {@code [REG]} token's embedding, or −1. */
    public int reg() {
        return reg;
    }

    public Residual inputEmbedding() {
        return inputEmbedding;
    }

    public List<Layer> layers() {
        return layers;
    }

    public int finalNorm() {
        return finalNorm;
    }

    public Residual outputEmbedding() {
        return outputEmbedding;
    }

    /** The weight matrix {@code [out, in]} of {@code l} in a buffer of its own, for library GEMMs. */
    public synchronized FloatArray matrix(Lin l) {
        return matrices.computeIfAbsent(l, x -> {
            FloatArray m = new FloatArray(x.out() * x.in());
            for (int i = 0; i < m.getSize(); i++) {
                m.set(i, data.get(x.w() + i));
            }
            return m;
        });
    }

    /** The weight matrix {@code [out, in]} of {@code l} in FP16, for the FP16 GEMMs. */
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
        private int size;

        int add(float[] v) {
            int off = size;
            parts.add(v);
            size += v.length;
            return off;
        }

        Lin lin(Chronos2Weights.Linear l) {
            int w = add(l.weight().toFloatArray());
            int b = l.bias() == null ? -1 : add(l.bias().toFloatArray());
            return new Lin(w, b, l.in(), l.out());
        }

        Attention attention(Chronos2Weights w, Chronos2Weights.Attention a, Chronos2Config c) {
            int norm = add(a.norm().toFloatArray());
            Lin qkv = new Lin(add(w.qkv(a)), -1, c.dModel(), 3 * c.inner());
            return new Attention(norm, qkv, lin(a.o()));
        }

        Residual residual(Chronos2Weights.ResidualBlock r) {
            return new Residual(lin(r.hidden()), lin(r.output()), lin(r.residual()));
        }

        FloatArray build() {
            FloatArray f = new FloatArray(size);
            int off = 0;
            for (float[] v : parts) {
                for (int i = 0; i < v.length; i++) {
                    f.set(off + i, v[i]);
                }
                off += v.length;
            }
            return f;
        }
    }
}
