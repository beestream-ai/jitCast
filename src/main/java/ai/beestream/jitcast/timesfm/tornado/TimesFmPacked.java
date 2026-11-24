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
package ai.beestream.jitcast.timesfm.tornado;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.timesfm.TimesFmConfig;
import ai.beestream.jitcast.timesfm.TimesFmWeights;
import ai.beestream.jitcast.tornado.Kernels;

import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Every TimesFM 2.5 weight in one device buffer, addressed by element offsets, with each layer's query
 * scale ({@code 1.442695041/√d · softplus(w)}) and RoPE's inverse timescales precomputed. The library
 * GEMMs (cuBLAS) cannot address an offset into that buffer, so each matrix also has its own FP32 or FP16
 * copy, made on first use ({@link #matrix}, {@link #halfMatrix}).
 */
public final class TimesFmPacked {

    /**
     * A linear layer in the buffer.
     *
     * @param w the offset of {@code W [out, in]}
     * @param b the offset of the bias, or −1
     * @param in the input width
     * @param out the output width
     */
    public record Lin(int w, int b, int in, int out) {}

    /**
     * {@code output(SiLU(hidden(x))) + residual(x)}.
     *
     * @param hidden the first layer
     * @param output the second layer
     * @param residual the linear shortcut
     */
    public record Residual(Lin hidden, Lin output, Lin residual) {}

    /**
     * One transformer layer; norms and scales are offsets into the buffer.
     *
     * @param preAttn the RMSNorm scale before attention
     * @param qkv the fused projection
     * @param qNorm the per-head query RMSNorm scale
     * @param kNorm the per-head key RMSNorm scale
     * @param qScale the per-dimension query scale, softplus applied
     * @param out the attention output projection
     * @param postAttn the RMSNorm scale of the attention output
     * @param preFf the RMSNorm scale before the feed-forward layer
     * @param ff0 the feed-forward input layer
     * @param ff1 the feed-forward output layer
     * @param postFf the RMSNorm scale of the feed-forward output
     */
    public record Layer(int preAttn, Lin qkv, int qNorm, int kNorm, int qScale, Lin out, int postAttn, int preFf, Lin ff0, Lin ff1,
            int postFf) {}

    private static final double LOG2E = 1.442695041;

    private final FloatArray data;
    private final Residual tokenizer;
    private final List<Layer> layers;
    private final Residual pointHead;
    private final Residual quantileHead;
    private final int invTimescale;
    private final Map<Lin, FloatArray> matrices = new IdentityHashMap<>();
    private final Map<Lin, HalfFloatArray> halfMatrices = new IdentityHashMap<>();

    private TimesFmPacked(FloatArray data, Residual tokenizer, List<Layer> layers, Residual pointHead, Residual quantileHead,
            int invTimescale) {
        this.data = data;
        this.tokenizer = tokenizer;
        this.layers = layers;
        this.pointHead = pointHead;
        this.quantileHead = quantileHead;
        this.invTimescale = invTimescale;
    }

    /**
     * {@return the weights packed into one buffer}
     *
     * @param w the checkpoint
     */
    public static TimesFmPacked pack(TimesFmWeights w) {
        TimesFmConfig c = w.config();
        int hd = c.headDim();
        Packer p = new Packer();
        Residual tokenizer = p.residual(w.tokenizer());
        List<Layer> layers = new ArrayList<>();
        for (TimesFmWeights.Layer l : w.layers()) {
            float[] pds = l.perDimScale().toFloatArray();
            float[] scale = new float[hd];
            for (int i = 0; i < hd; i++) {
                double x = pds[i];
                scale[i] = (float) (LOG2E / Math.sqrt(hd) * (x > 20 ? x : Math.log1p(Math.exp(x))));
            }
            layers.add(new Layer(p.add(l.preAttnNorm().toFloatArray()), p.lin(l.qkv()), p.add(l.queryNorm().toFloatArray()),
                    p.add(l.keyNorm().toFloatArray()), p.add(scale), p.lin(l.out()), p.add(l.postAttnNorm().toFloatArray()),
                    p.add(l.preFfNorm().toFloatArray()), p.lin(l.ff0()), p.lin(l.ff1()), p.add(l.postFfNorm().toFloatArray())));
        }
        Residual point = p.residual(w.pointHead());
        Residual quantiles = p.residual(w.quantileHead());
        float[] inv = new float[hd / 2];
        for (int i = 0; i < inv.length; i++) {
            inv[i] = (float) (1.0 / Math.pow(10000.0, 2.0 * i / hd));
        }
        int invTs = p.add(inv);
        return new TimesFmPacked(p.build(), tokenizer, List.copyOf(layers), point, quantiles, invTs);
    }

    /** {@return the device buffer} */
    public FloatArray data() {
        return data;
    }

    /** {@return the tokenizer} */
    public Residual tokenizer() {
        return tokenizer;
    }

    /** {@return the transformer layers} */
    public List<Layer> layers() {
        return layers;
    }

    /** {@return the point head} */
    public Residual pointHead() {
        return pointHead;
    }

    /** {@return the continuous quantile head} */
    public Residual quantileHead() {
        return quantileHead;
    }

    /** {@return the offset of RoPE's inverse timescales, {@code headDim / 2} of them} */
    public int invTimescale() {
        return invTimescale;
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
        private int size;

        int add(float[] v) {
            int off = size;
            parts.add(v);
            size += v.length;
            return off;
        }

        Lin lin(TimesFmWeights.Linear l) {
            int w = add(l.weight().toFloatArray());
            int b = l.bias() == null ? -1 : add(l.bias().toFloatArray());
            return new Lin(w, b, l.in(), l.out());
        }

        Residual residual(TimesFmWeights.ResidualBlock r) {
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
