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
package ai.beestream.jitcast.toto.tornado;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.toto.TotoConfig;
import ai.beestream.jitcast.toto.TotoForward;
import ai.beestream.jitcast.toto.TotoWeights;
import ai.beestream.jitcast.tornado.Kernels;

import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Toto 2.0's weights in one device buffer, with every constant of the unit scaling folded in, so that the
 * device runs plain linear layers:
 *
 * <ul>
 * <li>each linear layer's {@code 1/√fan_in} (the output head's readouts {@code 1/fan_in}), bias included;</li>
 * <li>the residual MLPs' {@code 1/√2} and the unit-scaled SiLU's forward scale (into the output layer's
 * weights, the SiLU itself being plain);</li>
 * <li>the query's per-dimension scale and the attention's {@code 1/head_dim} (into the query rows);</li>
 * <li>each branch's {@code τ/√(1 + τ²)} (into the attention output and the second feed-forward layer); the
 * residual stream's {@code 1/√(1 + τ²)} is a kernel argument ({@link Layer#attnKeep()},
 * {@link Layer#mlpKeep()}).</li>
 * </ul>
 * The gated feed-forward's input rows are reordered to {@code [SiLU input | gate]}, the order of the shared
 * SwiGLU kernel, and its output is shifted down by {@link #GLU_SHIFT}. A vector of ones serves the weightless RMSNorms.
 */
public final class TotoPacked {

    /**
     * A linear layer at offsets in {@link #data()}.
     *
     * @param w the weight, {@code [out, in]}
     * @param b the bias, or −1
     * @param in the input width
     * @param out the output width
     * @param gain a power of two near the weights' RMS when that is below 1/16, else 1: the FP16 copy holds
     *        the weights divided by it, so that small weights (the output head's readouts) stay out of the FP16
     *        subnormals, and the FP16 GEMM multiplies it back
     */
    public record Lin(int w, int b, int in, int out, float gain) {}

    /**
     * A residual MLP: {@code linear2(silu(linear1(x))) + skip(x)}, scales folded in.
     *
     * @param linear1 the hidden layer
     * @param linear2 the output layer
     * @param skip the skip projection
     */
    public record Mlp(Lin linear1, Lin linear2, Lin skip) {}

    /**
     * One layer.
     *
     * @param inProj the query, key and value projection
     * @param outProj the attention output, scaled by the branch weight
     * @param attnKeep the residual stream's factor around attention
     * @param fc1 the feed-forward input, {@code [SiLU input | gate]}
     * @param fc2 the feed-forward output, scaled by the branch weight
     * @param mlpKeep the residual stream's factor around the feed-forward
     */
    public record Layer(Lin inProj, Lin outProj, float attnKeep, Lin fc1, Lin fc2, float mlpKeep) {}

    /**
     * The gated feed-forward's output is scaled by this power of two (through the gate rows) and the second
     * layer's weights by its inverse: exact in FP32, and it keeps that output, which reaches several hundred
     * thousand, within FP16's range.
     */
    static final double GLU_SHIFT = 1.0 / 256;

    private final FloatArray data;
    private final Mlp patchProj;
    private final List<Layer> layers;
    private final Mlp head;
    private final int ones;
    private final Map<Lin, FloatArray> matrices = new IdentityHashMap<>();
    private final Map<Lin, HalfFloatArray> halfMatrices = new IdentityHashMap<>();

    private TotoPacked(FloatArray data, Mlp patchProj, List<Layer> layers, Mlp head, int ones) {
        this.data = data;
        this.patchProj = patchProj;
        this.layers = layers;
        this.head = head;
        this.ones = ones;
    }

    /**
     * {@return a checkpoint's weights, packed}
     *
     * @param w the checkpoint
     */
    public static TotoPacked pack(TotoWeights w) {
        TotoConfig c = w.config();
        int d = c.dModel(), hd = c.headDim(), ff = c.dFf();
        Packer p = new Packer();
        Mlp patch = p.mlp(w.patchProj(), false);
        List<Layer> layers = new ArrayList<>();
        for (TotoWeights.Layer l : w.layers()) {
            float[] pds = l.perDimScale().toFloatArray();
            double s = 1.0 / Math.sqrt(d), attn = TotoForward.attentionScale(c);
            double[] rowScale = new double[3 * d];
            for (int r = 0; r < 3 * d; r++) {
                double x = pds[r % hd];
                rowScale[r] = r < d ? s * (x > 20 ? x : Math.log1p(Math.exp(x))) / Math.log(2.0) * attn : s;
            }
            Lin in = p.lin(l.inProj(), rowScale, rowScale);
            double ka = 1.0 / Math.sqrt(1 + (double) l.attnTau() * l.attnTau());
            double km = 1.0 / Math.sqrt(1 + (double) l.mlpTau() * l.mlpTau());
            Lin out = p.lin(l.outProj(), uniform(d, l.attnTau() * ka / Math.sqrt(d)), uniform(d, l.attnTau() * ka / Math.sqrt(d)));
            float[] fc1 = l.fc1().weight().toFloatArray();
            float[] swapped = new float[fc1.length];
            System.arraycopy(fc1, ff * d, swapped, 0, ff * d);
            System.arraycopy(fc1, 0, swapped, ff * d, ff * d);
            double[] f1Scale = uniform(2 * ff, 1.0 / Math.sqrt(d));
            for (int r = ff; r < 2 * ff; r++) {
                f1Scale[r] *= GLU_SHIFT;
            }
            Lin f1 = p.lin(swapped, null, d, 2 * ff, f1Scale, null);
            Lin f2 = p.lin(l.fc2(), uniform(d, l.mlpTau() * km / Math.sqrt(ff) / GLU_SHIFT), null);
            layers.add(new Layer(in, out, (float) ka, f1, f2, (float) km));
        }
        Mlp head = p.mlp(w.head(), true);
        float[] one = new float[d];
        Arrays.fill(one, 1f);
        int ones = p.add(one);
        return new TotoPacked(p.build(), patch, List.copyOf(layers), head, ones);
    }

    private static double[] uniform(int n, double v) {
        double[] a = new double[n];
        Arrays.fill(a, v);
        return a;
    }

    /** {@return the packed buffer} */
    public FloatArray data() {
        return data;
    }

    /** {@return the patch embedding} */
    public Mlp patchProj() {
        return patchProj;
    }

    /** {@return the layers} */
    public List<Layer> layers() {
        return layers;
    }

    /** {@return the output head} */
    public Mlp head() {
        return head;
    }

    /** {@return the offset of {@code d_model} ones, the weightless norms' weight} */
    public int ones() {
        return ones;
    }

    /**
     * {@return a layer's weight as its own FP32 buffer}, for cuBLAS
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
     * {@return a layer's weight divided by its {@link Lin#gain()}, as its own FP16 buffer}
     *
     * @param l the layer
     */
    public synchronized HalfFloatArray halfMatrix(Lin l) {
        return halfMatrices.computeIfAbsent(l, x -> {
            HalfFloatArray m = new HalfFloatArray(x.out() * x.in());
            for (int i = 0; i < m.getSize(); i++) {
                m.set(i, new HalfFloat(Math.clamp(data.get(x.w() + i) / x.gain(), -Kernels.HALF_MAX, Kernels.HALF_MAX)));
            }
            return m;
        });
    }

    private static final class Packer {
        private final List<float[]> parts = new ArrayList<>();
        private int size;

        int add(float[] v) {
            int off = size;
            parts.add(v);
            size += v.length;
            return off;
        }

        /** A linear layer with each output row's weights scaled by {@code wScale[row]} and bias by {@code bScale[row]}. */
        Lin lin(float[] weight, float[] bias, int in, int out, double[] wScale, double[] bScale) {
            float[] ws = new float[weight.length];
            for (int r = 0; r < out; r++) {
                for (int i = 0; i < in; i++) {
                    ws[r * in + i] = (float) (weight[r * in + i] * wScale[r]);
                }
            }
            double ss = 0;
            for (float v : ws) {
                ss += (double) v * v;
            }
            double rms = Math.sqrt(ss / ws.length);
            float gain = rms >= 1.0 / 16 || rms == 0 ? 1f : (float) Math.pow(2, Math.round(Math.log(rms) / Math.log(2)));
            int w = add(ws);
            int b = -1;
            if (bias != null) {
                float[] bs = new float[out];
                for (int r = 0; r < out; r++) {
                    bs[r] = (float) (bias[r] * bScale[r]);
                }
                b = add(bs);
            }
            return new Lin(w, b, in, out, gain);
        }

        Lin lin(TotoWeights.Linear l, double[] wScale, double[] bScale) {
            return lin(l.weight().toFloatArray(), l.bias() == null ? null : l.bias().toFloatArray(), l.in(), l.out(), wScale, bScale);
        }

        Mlp mlp(TotoWeights.ResidualMlp m, boolean readout) {
            TotoWeights.Linear l1 = m.linear1(), l2 = m.linear2(), sk = m.skip();
            Lin a = lin(l1, uniform(l1.out(), 1.0 / Math.sqrt(l1.in())), uniform(l1.out(), 1.0 / Math.sqrt(l1.in())));
            double s2 = (readout ? 1.0 / l2.in() : 1.0 / Math.sqrt(l2.in())) * TotoForward.RESIDUAL_MLP_SCALE;
            double s3 = (readout ? 1.0 / sk.in() : 1.0 / Math.sqrt(sk.in())) * TotoForward.RESIDUAL_MLP_SCALE;
            Lin b = lin(l2, uniform(l2.out(), s2 * TotoForward.SILU_SCALE), uniform(l2.out(), s2));
            Lin skip = lin(sk, uniform(sk.out(), s3), uniform(sk.out(), s3));
            return new Mlp(a, b, skip);
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
