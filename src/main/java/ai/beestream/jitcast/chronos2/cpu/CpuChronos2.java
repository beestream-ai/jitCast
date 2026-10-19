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
package ai.beestream.jitcast.chronos2.cpu;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import ai.beestream.jitcast.chronos2.Chronos2Config;
import ai.beestream.jitcast.chronos2.Chronos2Weights;
import ai.beestream.jitcast.chronos2.ModelForward;
import ai.beestream.jitcast.chronos2.ModelInput;
import ai.beestream.jitcast.chronos2.ModelLayout;
import ai.beestream.jitcast.chronos2.ModelPrep;
import ai.beestream.jitcast.cpu.Ops;

/**
 * The FP32 CPU reference of {@code Chronos2Model.forward}, the oracle for the device kernels: the
 * instance norm, the input patch embedding, the encoder (time attention with RoPE, group attention
 * across the rows of a group, the ReLU feed-forward layer) over the {@link ModelLayout} tokens, the
 * output patch embedding and the inverse norm. Activations are row-major
 * {@code [rows · tokens · D]}; reductions accumulate in double. As in the reference, attention
 * scores are not scaled by {@code 1/√d_kv}.
 */
public final class CpuChronos2 implements ModelForward {

    /** Receives each stage's activation, named as the golden files name them. */
    @FunctionalInterface
    public interface StageSink {
        void stage(String name, float[] values);
    }

    record Linear(float[] w, float[] b, int in, int out) {
        static Linear of(Chronos2Weights.Linear l) {
            return new Linear(l.weight().toFloatArray(), l.bias() == null ? null : l.bias().toFloatArray(), l.in(), l.out());
        }

        float[] apply(float[] x, int rows) {
            return Ops.linear(x, rows, w, b, in, out);
        }
    }

    record Residual(Linear hidden, Linear output, Linear residual) {
        static Residual of(Chronos2Weights.ResidualBlock r) {
            return new Residual(Linear.of(r.hidden()), Linear.of(r.output()), Linear.of(r.residual()));
        }

        /** {@code output(ReLU(hidden(x))) + residual(x)}. */
        float[] apply(float[] x, int rows) {
            float[] h = hidden.apply(x, rows);
            Ops.reluInPlace(h);
            float[] y = output.apply(h, rows);
            Ops.addInPlace(y, residual.apply(x, rows));
            return y;
        }
    }

    record Attention(float[] norm, Linear qkv, Linear o) {}

    record Layer(Attention time, Attention group, float[] mlpNorm, Linear wi, Linear wo) {}

    private final Chronos2Config config;
    private final Residual inputEmbedding;
    private final float[] reg;
    private final List<Layer> layers = new ArrayList<>();
    private final float[] finalNorm;
    private final Residual outputEmbedding;
    /** RoPE inverse frequencies, one per pair of head dimensions. */
    private final float[] invFreq;

    public CpuChronos2(Chronos2Weights w) {
        config = w.config();
        inputEmbedding = Residual.of(w.inputEmbedding());
        reg = config.useRegToken() ? w.regToken() : null;
        int d = config.dModel(), inner = config.inner();
        for (Chronos2Weights.Layer l : w.layers()) {
            layers.add(new Layer(attention(w, l.time(), d, inner), attention(w, l.group(), d, inner), l.mlpNorm().toFloatArray(),
                    Linear.of(l.wi()), Linear.of(l.wo())));
        }
        finalNorm = w.finalNorm().toFloatArray();
        outputEmbedding = Residual.of(w.outputEmbedding());
        invFreq = inverseFrequencies(config);
    }

    private static Attention attention(Chronos2Weights w, Chronos2Weights.Attention a, int d, int inner) {
        return new Attention(a.norm().toFloatArray(), new Linear(w.qkv(a), null, d, 3 * inner), Linear.of(a.o()));
    }

    /** RoPE's inverse frequencies, {@code 1 / θ^(2i/d_kv)}, as {@code Chronos2RotaryEmbedding} builds them in float. */
    public static float[] inverseFrequencies(Chronos2Config c) {
        int dh = c.dKv();
        float[] f = new float[dh / 2];
        for (int i = 0; i < f.length; i++) {
            float e = (float) (2 * i) / (float) dh;
            f[i] = 1f / (float) Math.pow(c.ropeTheta(), e);
        }
        return f;
    }

    public Chronos2Config config() {
        return config;
    }

    @Override
    public float[][][] forward(ModelInput input) {
        return forward(input, null);
    }

    public float[][][] forward(ModelInput input, StageSink sink) {
        Chronos2Config c = config;
        ModelLayout lay = ModelLayout.of(c, input);
        int r = lay.rows(), d = c.dModel();
        int nc = lay.contextPatches(), h = lay.outputPatches(), n = lay.tokens(), ne = nc + h, fs = lay.futureStart();

        // instance norm and the patch tokens, on the host
        ModelPrep prep = ModelPrep.of(c, lay);
        emit(sink, "loc", prep.loc());
        emit(sink, "scale", prep.scale());

        // the embedding of every context and future patch; [REG] between them
        float[] embed = inputEmbedding.apply(prep.tokens(), r * ne);
        float[] x = new float[r * n * d];
        float[] ec = new float[r * nc * d], ef = new float[r * h * d];
        for (int i = 0; i < r; i++) {
            System.arraycopy(embed, i * ne * d, x, i * n * d, nc * d);
            if (reg != null) {
                System.arraycopy(reg, 0, x, (i * n + nc) * d, d);
            }
            System.arraycopy(embed, (i * ne + nc) * d, x, (i * n + fs) * d, h * d);
            System.arraycopy(embed, i * ne * d, ec, i * nc * d, nc * d);
            System.arraycopy(embed, (i * ne + nc) * d, ef, i * h * d, h * d);
        }
        emit(sink, "embed_context", ec);
        emit(sink, "embed_future", ef);

        float eps = c.layerNormEps();
        for (int li = 0; li < layers.size(); li++) {
            Layer layer = layers.get(li);
            float[] hn = Ops.rmsNorm(x, r * n, d, layer.time().norm(), eps);
            Ops.addInPlace(x, layer.time().o().apply(timeAttention(lay, layer.time().qkv().apply(hn, r * n)), r * n));
            hn = Ops.rmsNorm(x, r * n, d, layer.group().norm(), eps);
            Ops.addInPlace(x, layer.group().o().apply(groupAttention(lay, layer.group().qkv().apply(hn, r * n)), r * n));
            hn = Ops.rmsNorm(x, r * n, d, layer.mlpNorm(), eps);
            float[] act = layer.wi().apply(hn, r * n);
            Ops.reluInPlace(act);
            Ops.addInPlace(x, layer.wo().apply(act, r * n));
            emit(sink, "layer" + li, x);
        }
        float[] y = Ops.rmsNorm(x, r * n, d, finalNorm, eps);
        emit(sink, "final_norm", y);

        // the output patch embedding of the future tokens
        float[] dec = new float[r * h * d];
        for (int i = 0; i < r; i++) {
            System.arraycopy(y, (i * n + fs) * d, dec, i * h * d, h * d);
        }
        float[] raw = outputEmbedding.apply(dec, r * h);
        emit(sink, "head", raw);
        return prep.finish(c, lay, raw);
    }

    /** Multi-head attention along time, per row, with RoPE on q and k; every valid key in both directions. */
    private float[] timeAttention(ModelLayout lay, float[] qkv) {
        Chronos2Config c = config;
        int r = lay.rows(), n = lay.tokens(), inner = c.inner(), heads = c.numHeads(), dh = c.dKv();
        float[] q = new float[r * n * inner], k = new float[r * n * inner];
        for (int i = 0; i < r; i++) {
            for (int t = 0; t < n; t++) {
                int o = (i * n + t) * 3 * inner;
                for (int hh = 0; hh < heads; hh++) {
                    rope(qkv, o + hh * dh, q, (i * n + t) * inner + hh * dh, t);
                    rope(qkv, o + inner + hh * dh, k, (i * n + t) * inner + hh * dh, t);
                }
            }
        }
        float[] out = new float[r * n * inner];
        IntStream.range(0, r * heads).parallel().forEach(job -> {
            int i = job / heads, hh = job % heads;
            double[] p = new double[n];
            for (int t = 0; t < n; t++) {
                double max = Double.NEGATIVE_INFINITY;
                int qo = (i * n + t) * inner + hh * dh;
                for (int s = 0; s < n; s++) {
                    if (!lay.live(i, s)) {
                        p[s] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    int ko = (i * n + s) * inner + hh * dh;
                    double dot = 0;
                    for (int e = 0; e < dh; e++) {
                        dot += (double) q[qo + e] * k[ko + e];
                    }
                    p[s] = dot;
                    max = Math.max(max, dot);
                }
                softmaxValues(p, max, qkv, (i * n) * 3 * inner, 3 * inner, 2 * inner + hh * dh, dh, out, qo);
            }
        });
        return out;
    }

    /**
     * {@code x·cos + rotate_half(x)·sin} for one head vector: dimensions {@code e} and
     * {@code e + dh/2} rotate together by {@code pos · invFreq[e]}.
     */
    private void rope(float[] src, int so, float[] dst, int dOff, int pos) {
        int half = config.dKv() / 2;
        for (int e = 0; e < half; e++) {
            float f = (float) pos * invFreq[e];
            float cos = (float) Math.cos(f), sin = (float) Math.sin(f);
            float a = src[so + e], b = src[so + half + e];
            dst[dOff + e] = a * cos + -b * sin;
            dst[dOff + half + e] = b * cos + a * sin;
        }
    }

    /**
     * Multi-head attention across the rows of each group, per token slot; no positional encoding. A
     * query whose group has no valid key at its slot (a context patch nobody observed) gets zeros.
     */
    private float[] groupAttention(ModelLayout lay, float[] qkv) {
        Chronos2Config c = config;
        int r = lay.rows(), n = lay.tokens(), inner = c.inner(), heads = c.numHeads(), dh = c.dKv();
        int[][] members = lay.members();
        float[] out = new float[r * n * inner];
        IntStream.range(0, n).parallel().forEach(t -> {
            for (int i = 0; i < r; i++) {
                int[] m = members[i];
                double[] p = new double[m.length];
                for (int hh = 0; hh < heads; hh++) {
                    int qo = (i * n + t) * 3 * inner + hh * dh;
                    double max = Double.NEGATIVE_INFINITY;
                    for (int jj = 0; jj < m.length; jj++) {
                        int j = m[jj];
                        if (!lay.live(j, t)) {
                            p[jj] = Double.NEGATIVE_INFINITY;
                            continue;
                        }
                        int ko = (j * n + t) * 3 * inner + inner + hh * dh;
                        double dot = 0;
                        for (int e = 0; e < dh; e++) {
                            dot += (double) qkv[qo + e] * qkv[ko + e];
                        }
                        p[jj] = dot;
                        max = Math.max(max, dot);
                    }
                    int oo = (i * n + t) * inner + hh * dh;
                    if (max == Double.NEGATIVE_INFINITY) {
                        continue;
                    }
                    double sum = 0;
                    for (int jj = 0; jj < m.length; jj++) {
                        p[jj] = p[jj] == Double.NEGATIVE_INFINITY ? 0 : Math.exp(p[jj] - max);
                        sum += p[jj];
                    }
                    for (int e = 0; e < dh; e++) {
                        double acc = 0;
                        for (int jj = 0; jj < m.length; jj++) {
                            if (p[jj] != 0) {
                                acc += p[jj] * qkv[(m[jj] * n + t) * 3 * inner + 2 * inner + hh * dh + e];
                            }
                        }
                        out[oo + e] = (float) (acc / sum);
                    }
                }
            }
        });
        return out;
    }

    /**
     * The softmax of the scores {@code p} (masked keys at −∞, {@code max} their maximum) applied to
     * the value vectors at {@code base + s·stride + vOff}, written to {@code out[oo ..]}.
     */
    private static void softmaxValues(double[] p, double max, float[] qkv, int base, int stride, int vOff, int dh, float[] out,
            int oo) {
        double sum = 0;
        for (int s = 0; s < p.length; s++) {
            p[s] = p[s] == Double.NEGATIVE_INFINITY ? 0 : Math.exp(p[s] - max);
            sum += p[s];
        }
        for (int e = 0; e < dh; e++) {
            double acc = 0;
            for (int s = 0; s < p.length; s++) {
                if (p[s] != 0) {
                    acc += p[s] * qkv[base + s * stride + vOff + e];
                }
            }
            out[oo + e] = (float) (acc / sum);
        }
    }

    private static void emit(StageSink sink, String name, float[] values) {
        if (sink != null) {
            sink.stage(name, values.clone());
        }
    }
}
