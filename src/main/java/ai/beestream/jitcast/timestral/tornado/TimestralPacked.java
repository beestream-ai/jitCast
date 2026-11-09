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
package ai.beestream.jitcast.timestral.tornado;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.timestral.TimestralConfig;
import ai.beestream.jitcast.timestral.TimestralWeights;
import ai.beestream.jitcast.tornado.Kernels;

import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Every Timestral weight in one device buffer, addressed by element offsets: the q|k|v projections
 * in heads-major order and the MLP input with its SiLU half first ({@link TimestralWeights#qkv},
 * {@link TimestralWeights#mlpUp}), everything else as stored. The library GEMMs (cuBLAS) cannot
 * address an offset into that buffer, so each matrix also has its own FP32 or FP16 copy, made on
 * first use ({@link #matrix}, {@link #halfMatrix}).
 */
public final class TimestralPacked {

    /** A linear layer in the buffer: {@code W [out, in]} at {@code w}, the bias at {@code b} or −1. */
    public record Lin(int w, int b, int in, int out) {}

    public record Residual(Lin hidden, Lin output, Lin residual) {}

    public record Attention(boolean time, int norm, Lin qkv, Lin gate, Lin wo) {}

    public record Layer(List<Attention> attentions, int mlpNorm, Lin up, Lin down) {}

    private final FloatArray data;
    private final Residual embedding;
    private final int forecastQuery;
    private final List<Layer> layers;
    private final int outNorm;
    private final Residual head;
    private final Map<Lin, FloatArray> matrices = new IdentityHashMap<>();
    private final Map<Lin, HalfFloatArray> halfMatrices = new IdentityHashMap<>();

    private TimestralPacked(FloatArray data, Residual embedding, int forecastQuery, List<Layer> layers, int outNorm,
            Residual head) {
        this.data = data;
        this.embedding = embedding;
        this.forecastQuery = forecastQuery;
        this.layers = layers;
        this.outNorm = outNorm;
        this.head = head;
    }

    public static TimestralPacked pack(TimestralWeights w) {
        TimestralConfig c = w.config();
        int d = c.dModel();
        Packer p = new Packer();
        Residual embed = p.residual(w.patchEmbedding());
        int query = p.add(w.forecastQuery().toFloatArray());
        List<Layer> layers = new ArrayList<>();
        for (TimestralWeights.Layer l : w.layers()) {
            List<Attention> attns = new ArrayList<>();
            for (TimestralWeights.Attention a : l.attentions()) {
                int norm = p.add(a.norm().toFloatArray());
                Lin qkv = new Lin(p.add(w.qkv(a)), -1, d, 3 * d);
                Lin gate = a.gate() == null ? null : p.lin(a.gate());
                attns.add(new Attention(a.time(), norm, qkv, gate, p.lin(a.wo())));
            }
            int mlpNorm = p.add(l.mlpNorm().toFloatArray());
            Lin up = new Lin(p.add(w.mlpUp(l)), -1, d, 2 * c.mlpHiddenDim());
            layers.add(new Layer(List.copyOf(attns), mlpNorm, up, p.lin(l.w3())));
        }
        int outNorm = p.add(w.outNorm().toFloatArray());
        Residual head = p.residual(w.head());
        return new TimestralPacked(p.build(), embed, query, List.copyOf(layers), outNorm, head);
    }

    public FloatArray data() {
        return data;
    }

    public Residual embedding() {
        return embedding;
    }

    public int forecastQuery() {
        return forecastQuery;
    }

    public List<Layer> layers() {
        return layers;
    }

    public int outNorm() {
        return outNorm;
    }

    public Residual head() {
        return head;
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

        Lin lin(TimestralWeights.Linear l) {
            int w = add(l.weight().toFloatArray());
            int b = l.bias() == null ? -1 : add(l.bias().toFloatArray());
            return new Lin(w, b, l.in(), l.out());
        }

        Residual residual(TimestralWeights.ResidualBlock r) {
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
