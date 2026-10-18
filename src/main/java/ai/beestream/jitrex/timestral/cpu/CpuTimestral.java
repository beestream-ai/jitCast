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
package ai.beestream.jitrex.timestral.cpu;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import ai.beestream.jitrex.cpu.Ops;
import ai.beestream.jitrex.timestral.BlockForward;
import ai.beestream.jitrex.timestral.BlockInput;
import ai.beestream.jitrex.timestral.BlockLayout;
import ai.beestream.jitrex.timestral.BlockPrep;
import ai.beestream.jitrex.timestral.Quantiles;
import ai.beestream.jitrex.timestral.TimestralConfig;
import ai.beestream.jitrex.timestral.TimestralWeights;

/**
 * The FP32 CPU reference of {@code TimestralModel.predict_block}, the oracle for the device
 * kernels: the scaler, the patch embedding, the time/variate transformer over the
 * {@link BlockLayout} tokens, the monotone quantile head and the denormalization. Activations are
 * row-major {@code [rows · tokens · D]}; reductions accumulate in double.
 */
public final class CpuTimestral implements BlockForward {

    /** Receives each stage's activation, named as the golden files name them. */
    @FunctionalInterface
    public interface StageSink {
        void stage(String name, float[] values);
    }

    record Linear(float[] w, float[] b, int in, int out) {
        static Linear of(TimestralWeights.Linear l) {
            return l == null ? null : new Linear(l.weight().toFloatArray(), l.bias() == null ? null : l.bias().toFloatArray(),
                    l.in(), l.out());
        }

        float[] apply(float[] x, int rows) {
            return Ops.linear(x, rows, w, b, in, out);
        }
    }

    record Residual(Linear hidden, Linear output, Linear residual) {
        static Residual of(TimestralWeights.ResidualBlock r) {
            return new Residual(Linear.of(r.hidden()), Linear.of(r.output()), Linear.of(r.residual()));
        }

        /** {@code output(SiLU(hidden(x))) + residual(x)}. */
        float[] apply(float[] x, int rows) {
            float[] h = hidden.apply(x, rows);
            Ops.siluInPlace(h);
            float[] y = output.apply(h, rows);
            Ops.addInPlace(y, residual.apply(x, rows));
            return y;
        }
    }

    record Attention(boolean time, float[] norm, Linear qkv, Linear gate, Linear wo) {}

    record Layer(List<Attention> attentions, float[] mlpNorm, Linear up, Linear down) {}

    private final TimestralConfig config;
    private final Residual embedding;
    private final float[] forecastQuery;
    private final List<Layer> layers = new ArrayList<>();
    private final float[] outNorm;
    private final Residual head;
    private final Quantiles quantiles;
    /** RoPE inverse frequencies for the first {@code headDim/2} dimensions, 0 past {@code ropeDim/2}. */
    private final float[] invFreq;

    public CpuTimestral(TimestralWeights w) {
        config = w.config();
        embedding = Residual.of(w.patchEmbedding());
        forecastQuery = w.forecastQuery().toFloatArray();
        int d = config.dModel();
        for (TimestralWeights.Layer l : w.layers()) {
            List<Attention> attns = new ArrayList<>();
            for (TimestralWeights.Attention a : l.attentions()) {
                attns.add(new Attention(a.time(), a.norm().toFloatArray(), new Linear(w.qkv(a), null, d, 3 * d),
                        Linear.of(a.gate()), Linear.of(a.wo())));
            }
            layers.add(new Layer(List.copyOf(attns), l.mlpNorm().toFloatArray(),
                    new Linear(w.mlpUp(l), null, d, 2 * config.mlpHiddenDim()), Linear.of(l.w3())));
        }
        outNorm = w.outNorm().toFloatArray();
        head = Residual.of(w.head());
        quantiles = new Quantiles(config);
        invFreq = inverseFrequencies(config);
    }

    /** RoPE's inverse frequencies, as {@code SinCosRoPE} builds them in float. */
    public static float[] inverseFrequencies(TimestralConfig c) {
        int dh = c.headDim(), rot = c.ropeDim();
        float[] f = new float[dh / 2];
        for (int i = 0; i < rot / 2; i++) {
            float e = (float) (2 * i) / (float) rot;
            f[i] = 1f / (float) Math.pow(c.ropeTheta(), e);
        }
        return f;
    }

    public TimestralConfig config() {
        return config;
    }

    @Override
    public float[][][] predictBlock(BlockInput block) {
        return predictBlock(block, null);
    }

    public float[][][] predictBlock(BlockInput block, StageSink sink) {
        TimestralConfig c = config;
        BlockLayout lay = BlockLayout.of(c, block);
        int r = lay.rows(), d = c.dModel();
        int enc = lay.encoderTokens(), h = lay.horizonPatches(), n = lay.tokens();

        // scaler (known covariates use whole-row statistics) and the patch tokens, on the host
        BlockPrep prep = BlockPrep.of(c, lay);
        emit(sink, "scaled", prep.scaled());
        emit(sink, "loc", prep.loc());
        emit(sink, "scale", prep.scale());

        // patch embedding of [values, missing] per encoder token; the decoder tokens are the forecast query
        float[] embed = embedding.apply(prep.tokens(), r * enc);
        emit(sink, "embed", embed);
        float[] x = new float[r * n * d];
        for (int i = 0; i < r; i++) {
            System.arraycopy(embed, i * enc * d, x, i * n * d, enc * d);
            for (int t = enc; t < n; t++) {
                System.arraycopy(forecastQuery, 0, x, (i * n + t) * d, d);
            }
        }

        for (int li = 0; li < layers.size(); li++) {
            Layer layer = layers.get(li);
            for (Attention a : layer.attentions()) {
                float[] hn = Ops.rmsNorm(x, r * n, d, a.norm(), c.normEps());
                float[] qkv = a.qkv().apply(hn, r * n);
                float[] att = a.time() ? timeAttention(lay, qkv) : variateAttention(lay, qkv);
                if (a.gate() != null) {
                    float[] g = a.gate().apply(hn, r * n);
                    for (int i = 0; i < att.length; i++) {
                        att[i] = att[i] * (float) Ops.sigmoid(g[i]);
                    }
                }
                Ops.addInPlace(x, a.wo().apply(att, r * n));
            }
            float[] hn = Ops.rmsNorm(x, r * n, d, layer.mlpNorm(), c.normEps());
            int f = c.mlpHiddenDim();
            float[] up = layer.up().apply(hn, r * n);
            float[] act = new float[r * n * f];
            for (int t = 0; t < r * n; t++) {
                for (int j = 0; j < f; j++) {
                    act[t * f + j] = Ops.silu(up[t * 2 * f + j]) * up[t * 2 * f + f + j];
                }
            }
            Ops.addInPlace(x, layer.down().apply(act, r * n));
            emit(sink, "layer" + li, x);
        }
        float[] y = Ops.rmsNorm(x, r * n, d, outNorm, c.normEps());
        emit(sink, "out_norm", y);

        // the head on the decoder tokens: P quantile vectors per token, made monotone
        float[] dec = new float[r * h * d];
        for (int i = 0; i < r; i++) {
            System.arraycopy(y, (i * n + enc) * d, dec, i * h * d, h * d);
        }
        float[] raw = head.apply(dec, r * h);
        float[][][] out = prep.finish(c, lay, quantiles, raw);
        emit(sink, "head", raw);
        return out;
    }

    /** Multi-head attention along time, per row, with RoPE on q and k; q|k|v heads-major per token. */
    private float[] timeAttention(BlockLayout lay, float[] qkv) {
        TimestralConfig c = config;
        int r = lay.rows(), n = lay.tokens(), d = c.dModel(), heads = c.numHeads(), dh = c.headDim();
        float[] q = new float[r * n * d], k = new float[r * n * d];
        for (int i = 0; i < r; i++) {
            for (int t = 0; t < n; t++) {
                int pos = lay.position(i, t);
                int o = (i * n + t) * 3 * d;
                for (int hh = 0; hh < heads; hh++) {
                    rope(qkv, o + hh * dh, q, (i * n + t) * d + hh * dh, pos);
                    rope(qkv, o + d + hh * dh, k, (i * n + t) * d + hh * dh, pos);
                }
            }
        }
        float[] out = new float[r * n * d];
        double scale = 1.0 / Math.sqrt(dh);
        IntStream.range(0, r * heads).parallel().forEach(job -> {
            int i = job / heads, hh = job % heads;
            double[] p = new double[n];
            for (int t = 0; t < n; t++) {
                double max = Double.NEGATIVE_INFINITY;
                int qo = (i * n + t) * d + hh * dh;
                for (int s = 0; s < n; s++) {
                    if (!lay.timeAllowed(i, t, s)) {
                        p[s] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    int ko = (i * n + s) * d + hh * dh;
                    double dot = 0;
                    for (int e = 0; e < dh; e++) {
                        dot += (double) q[qo + e] * k[ko + e];
                    }
                    p[s] = dot * scale;
                    max = Math.max(max, p[s]);
                }
                double sum = 0;
                for (int s = 0; s < n; s++) {
                    p[s] = p[s] == Double.NEGATIVE_INFINITY ? 0 : Math.exp(p[s] - max);
                    sum += p[s];
                }
                for (int e = 0; e < dh; e++) {
                    double acc = 0;
                    for (int s = 0; s < n; s++) {
                        if (p[s] != 0) {
                            acc += p[s] * qkv[(i * n + s) * 3 * d + 2 * d + hh * dh + e];
                        }
                    }
                    out[qo + e] = (float) (acc / sum);
                }
            }
        });
        return out;
    }

    /**
     * {@code x·cos + rotate_half(x)·sin} for one head vector: dimensions {@code e} and
     * {@code e + dh/2} rotate together by {@code pos · invFreq[e]}; past {@code ropeDim/2} the
     * angle is 0 and the dimensions pass through.
     */
    private void rope(float[] src, int so, float[] dst, int dOff, int pos) {
        int half = config.headDim() / 2;
        for (int e = 0; e < half; e++) {
            float f = (float) pos * invFreq[e];
            float cos = (float) Math.cos(f), sin = (float) Math.sin(f);
            float a = src[so + e], b = src[so + half + e];
            dst[dOff + e] = a * cos + -b * sin;
            dst[dOff + half + e] = b * cos + a * sin;
        }
    }

    /** Multi-head attention across the rows of each series, per token slot; no positional encoding. */
    private float[] variateAttention(BlockLayout lay, float[] qkv) {
        TimestralConfig c = config;
        int r = lay.rows(), n = lay.tokens(), d = c.dModel(), heads = c.numHeads(), dh = c.headDim();
        Map<Integer, List<Integer>> byGroup = new HashMap<>();
        for (int i = 0; i < r; i++) {
            byGroup.computeIfAbsent(lay.group(i), g -> new ArrayList<>()).add(i);
        }
        int[][] members = new int[r][];
        for (int i = 0; i < r; i++) {
            members[i] = byGroup.get(lay.group(i)).stream().mapToInt(Integer::intValue).toArray();
        }
        float[] out = new float[r * n * d];
        double scale = 1.0 / Math.sqrt(dh);
        IntStream.range(0, n).parallel().forEach(t -> {
            for (int i = 0; i < r; i++) {
                int[] m = members[i];
                double[] p = new double[m.length];
                for (int hh = 0; hh < heads; hh++) {
                    int qo = (i * n + t) * 3 * d + hh * dh;
                    double max = Double.NEGATIVE_INFINITY;
                    for (int jj = 0; jj < m.length; jj++) {
                        int j = m[jj];
                        if (!lay.variateAllowed(t, i, j)) {
                            p[jj] = Double.NEGATIVE_INFINITY;
                            continue;
                        }
                        int ko = (j * n + t) * 3 * d + d + hh * dh;
                        double dot = 0;
                        for (int e = 0; e < dh; e++) {
                            dot += (double) qkv[qo + e] * qkv[ko + e];
                        }
                        p[jj] = dot * scale;
                        max = Math.max(max, p[jj]);
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
                                acc += p[jj] * qkv[(m[jj] * n + t) * 3 * d + 2 * d + hh * dh + e];
                            }
                        }
                        out[(i * n + t) * d + hh * dh + e] = (float) (acc / sum);
                    }
                }
            }
        });
        return out;
    }

    private static void emit(StageSink sink, String name, float[] values) {
        if (sink != null) {
            sink.stage(name, values.clone());
        }
    }
}
