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
package ai.beestream.jitcast.t0.cpu;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import ai.beestream.jitcast.cpu.Ops;
import ai.beestream.jitcast.t0.T0Config;
import ai.beestream.jitcast.t0.T0Forward;
import ai.beestream.jitcast.t0.T0Weights;

/**
 * The FP32 reference of t0's forward pass ({@code T0Forecaster.forward}), the oracle the device kernels are
 * tested against. The patch encoder embeds each patch's {@code [values ‖ time index ‖ validity]} and adds its
 * variate type's embedding. Each layer: RMSNorm, fused q|k|v, per-head RMSNorm of q and k, then either
 * attention across time (the rotary embedding with xPos scaling, causal for targets and past-only
 * covariates, bidirectional for known-future covariates) or attention across the variates of each series at
 * each patch (a known-future covariate reads only known-future covariates), the output projection plus the
 * residual, then a SwiGLU feed-forward plus the residual. A final RMSNorm, the residual-block decoder and the
 * quantile head (the first level, then cumulative softplus increments) give each patch's quantiles for the
 * next patch.
 */
public final class CpuT0 implements T0Forward {

    private record Block(float[] hw, float[] hb, float[] ow, float[] ob, float[] rw, float[] rb, int in, int hidden, int out) {}

    private record Layer(boolean group, float[] attnNorm, float[] qkvW, float[] qkvB, float[] qNorm, float[] kNorm,
            float[] outW, float[] outB, float[] mlpNorm, float[] upW, float[] upB, float[] downW, float[] downB) {}

    /**
     * Every stage of one forward pass, for the golden tests.
     *
     * @param embed the patch encoder's output, {@code [rows · patches · D]}
     * @param layers each layer's output
     * @param hidden the final RMSNorm's output
     * @param decoded the decoder's raw output, {@code [rows · patches · patchSize · quantiles]}
     * @param head the quantile head's output, the same shape
     */
    public record Stages(float[] embed, List<float[]> layers, float[] hidden, float[] decoded, float[] head) {}

    private final T0Config config;
    private final Block encoder;
    private final float[] typeEmbeddings;
    private final List<Layer> layers;
    private final float[] outNorm;
    private final Block decoder;
    private final float[] invFreq;
    private final float[] xposBase;

    /**
     * Copies the weights it reads to the heap.
     *
     * @param weights the checkpoint
     */
    public CpuT0(T0Weights weights) {
        this.config = weights.config();
        this.encoder = block(weights.patchEncoder());
        this.typeEmbeddings = weights.typeEmbeddings().toFloatArray();
        List<Layer> ls = new ArrayList<>();
        for (T0Weights.Layer l : weights.layers()) {
            ls.add(new Layer(l.group(), l.attnNorm().toFloatArray(), l.qkv().weight().toFloatArray(), l.qkv().bias().toFloatArray(),
                    l.queryNorm().toFloatArray(), l.keyNorm().toFloatArray(), l.out().weight().toFloatArray(),
                    l.out().bias().toFloatArray(), l.mlpNorm().toFloatArray(), l.up().weight().toFloatArray(),
                    l.up().bias().toFloatArray(), l.down().weight().toFloatArray(), l.down().bias().toFloatArray()));
        }
        this.layers = List.copyOf(ls);
        this.outNorm = weights.outNorm().toFloatArray();
        this.decoder = block(weights.decoder());
        this.invFreq = inverseFrequencies(config.headDim());
        this.xposBase = xposBase(config.headDim());
    }

    /** {@return the rotary embedding's per-pair frequencies, {@code 1 / θ^(2i/d)}, as float32} */
    public static float[] inverseFrequencies(int hd) {
        float[] f = new float[hd / 2];
        for (int i = 0; i < f.length; i++) {
            f[i] = 1f / (float) Math.pow(T0Config.ROPE_THETA, (float) (2 * i) / hd);
        }
        return f;
    }

    /** {@return the xPos per-pair base scales, {@code (2i + 0.4 d) / 1.4 d}, as float32} */
    public static float[] xposBase(int hd) {
        float[] s = new float[hd / 2];
        for (int i = 0; i < s.length; i++) {
            s[i] = ((float) (2 * i) + (float) (0.4 * hd)) / (float) (1.4 * hd);
        }
        return s;
    }

    /**
     * {@return the xPos exponent of a patch position: its distance from the window's centre over the scale base}
     *
     * @param s the patch position
     * @param patches the patches in the window
     */
    public static float xposPower(int s, int patches) {
        return ((float) s - (float) ((patches - 1) / 2)) / T0Config.XPOS_SCALE_BASE;
    }

    private static Block block(T0Weights.ResidualBlock b) {
        return new Block(b.hidden().weight().toFloatArray(), b.hidden().bias().toFloatArray(), b.output().weight().toFloatArray(),
                b.output().bias().toFloatArray(), b.residual().weight().toFloatArray(), b.residual().bias().toFloatArray(),
                b.hidden().in(), b.hidden().out(), b.output().out());
    }

    private static float[] residualBlock(float[] x, int rows, Block b) {
        float[] h = Ops.linear(x, rows, b.hw, b.hb, b.in, b.hidden);
        Ops.reluInPlace(h);
        float[] y = Ops.linear(h, rows, b.ow, b.ob, b.hidden, b.out);
        Ops.addInPlace(y, Ops.linear(x, rows, b.rw, b.rb, b.in, b.out));
        return y;
    }

    @Override
    public float[] forward(Input in, int from, int count) {
        float[] hidden = trunk(in, null);
        return head(slicePatches(hidden, in.rows(), in.patches(), from, count, config.embedDim()), in.rows() * count, null);
    }

    /**
     * {@return every stage of the forward pass, over all patches}
     *
     * @param in the inputs
     */
    public Stages stages(Input in) {
        List<float[]> collect = new ArrayList<>();
        float[] hidden = trunk(in, collect);
        float[][] decoded = new float[1][];
        float[] head = head(hidden, in.rows() * in.patches(), decoded);
        return new Stages(collect.get(0), List.copyOf(collect.subList(1, collect.size())), hidden, decoded[0], head);
    }

    static float[] slicePatches(float[] x, int rows, int patches, int from, int count, int width) {
        float[] out = new float[rows * count * width];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(x, (r * patches + from) * width, out, r * count * width, count * width);
        }
        return out;
    }

    /** The patch encoder and the layers, then the final RMSNorm. {@code collect} gets the embedding and each layer. */
    private float[] trunk(Input in, List<float[]> collect) {
        int rows = in.rows(), n = in.patches(), p = config.patchSize(), d = config.embedDim(), tokens = rows * n;
        float[] x = new float[tokens * 3 * p];
        for (int t = 0; t < tokens; t++) {
            for (int j = 0; j < p; j++) {
                x[t * 3 * p + j] = in.values()[t * p + j];
                x[t * 3 * p + p + j] = (float) j / p;
                x[t * 3 * p + 2 * p + j] = in.valid()[t * p + j];
            }
        }
        float[] h = residualBlock(x, tokens, encoder);
        for (int t = 0; t < tokens; t++) {
            int type = in.patchType()[t];
            for (int c = 0; c < d; c++) {
                h[t * d + c] += typeEmbeddings[type * d + c];
            }
        }
        if (collect != null) {
            collect.add(h.clone());
        }
        for (Layer l : layers) {
            layer(h, in, l);
            if (collect != null) {
                collect.add(h.clone());
            }
        }
        return Ops.rmsNorm(h, tokens, d, outNorm, T0Config.NORM_EPS);
    }

    private void layer(float[] h, Input in, Layer l) {
        int rows = in.rows(), n = in.patches(), d = config.embedDim(), tokens = rows * n;
        int heads = config.numHeads(), hd = config.headDim(), ff = config.mlpHiddenDim();
        float[] normed = Ops.rmsNorm(h, tokens, d, l.attnNorm, T0Config.NORM_EPS);
        float[] qkv = Ops.linear(normed, tokens, l.qkvW, l.qkvB, d, 3 * d);
        // Per-head RMSNorm of the queries and keys.
        for (int t = 0; t < tokens; t++) {
            for (int part = 0; part < 2; part++) {
                float[] scale = part == 0 ? l.qNorm : l.kNorm;
                for (int hh = 0; hh < heads; hh++) {
                    int o = t * 3 * d + part * d + hh * hd;
                    double ss = 0;
                    for (int e = 0; e < hd; e++) {
                        ss += (double) qkv[o + e] * qkv[o + e];
                    }
                    float inv = (float) (1.0 / Math.sqrt(ss / hd + T0Config.NORM_EPS));
                    for (int e = 0; e < hd; e++) {
                        qkv[o + e] = qkv[o + e] * inv * scale[e];
                    }
                }
            }
        }
        if (!l.group) {
            rotate(qkv, rows, n);
        }
        float[] attn = new float[tokens * d];
        if (l.group) {
            groupAttention(qkv, attn, in);
        } else {
            timeAttention(qkv, attn, in);
        }
        Ops.addInPlace(h, Ops.linear(attn, tokens, l.outW, l.outB, d, d));
        float[] up = Ops.linear(Ops.rmsNorm(h, tokens, d, l.mlpNorm, T0Config.NORM_EPS), tokens, l.upW, l.upB, d, 2 * ff);
        float[] act = new float[tokens * ff];
        for (int t = 0; t < tokens; t++) {
            for (int c = 0; c < ff; c++) {
                act[t * ff + c] = Ops.silu(up[t * 2 * ff + c]) * up[t * 2 * ff + ff + c];
            }
        }
        Ops.addInPlace(h, Ops.linear(act, tokens, l.downW, l.downB, ff, d));
    }

    /** The rotary embedding with xPos scaling on the queries ({@code scale}) and keys ({@code 1 / scale}). */
    private void rotate(float[] qkv, int rows, int n) {
        int d = config.embedDim(), heads = config.numHeads(), hd = config.headDim(), half = hd / 2;
        float[] cos = new float[n * half], sin = new float[n * half], sq = new float[n * hd], sk = new float[n * hd];
        for (int s = 0; s < n; s++) {
            float power = xposPower(s, n);
            for (int i = 0; i < half; i++) {
                float angle = (float) s * invFreq[i];
                cos[s * half + i] = (float) Math.cos(angle);
                sin[s * half + i] = (float) Math.sin(angle);
            }
            for (int e = 0; e < hd; e++) {
                int pair = config.xposHalves() ? e % half : e / 2;
                float sc = (float) Math.pow(xposBase[pair], power);
                sq[s * hd + e] = sc;
                sk[s * hd + e] = 1f / sc;
            }
        }
        for (int r = 0; r < rows; r++) {
            for (int s = 0; s < n; s++) {
                int t = r * n + s;
                for (int part = 0; part < 2; part++) {
                    float[] sc = part == 0 ? sq : sk;
                    for (int hh = 0; hh < heads; hh++) {
                        int o = t * 3 * d + part * d + hh * hd;
                        for (int i = 0; i < half; i++) {
                            float c = cos[s * half + i], si = sin[s * half + i];
                            float a = qkv[o + 2 * i], b = qkv[o + 2 * i + 1];
                            float s0 = sc[s * hd + 2 * i], s1 = sc[s * hd + 2 * i + 1];
                            qkv[o + 2 * i] = a * c * s0 + -b * si * s0;
                            qkv[o + 2 * i + 1] = b * c * s1 + a * si * s1;
                        }
                    }
                }
            }
        }
    }

    /** Attention along each row's patches: over the valid patches, causal except for known-future covariates. */
    private void timeAttention(float[] qkv, float[] out, Input in) {
        int rows = in.rows(), n = in.patches(), d = config.embedDim(), heads = config.numHeads(), hd = config.headDim();
        double scale = 1.0 / Math.sqrt(hd);
        IntStream.range(0, rows * heads).parallel().forEach(job -> {
            int r = job / heads, hh = job % heads;
            double[] w = new double[n];
            for (int i = 0; i < n; i++) {
                if (!in.patchValid()[r * n + i]) {
                    continue;
                }
                int qo = (r * n + i) * 3 * d + hh * hd;
                int last = in.future()[r] ? n - 1 : i;
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j <= last; j++) {
                    if (!in.patchValid()[r * n + j]) {
                        continue;
                    }
                    int ko = (r * n + j) * 3 * d + d + hh * hd;
                    double dot = 0;
                    for (int e = 0; e < hd; e++) {
                        dot += (double) qkv[qo + e] * qkv[ko + e];
                    }
                    w[j] = dot * scale;
                    max = Math.max(max, w[j]);
                }
                double sum = 0;
                for (int j = 0; j <= last; j++) {
                    if (in.patchValid()[r * n + j]) {
                        w[j] = Math.exp(w[j] - max);
                        sum += w[j];
                    }
                }
                int oo = (r * n + i) * d + hh * hd;
                for (int e = 0; e < hd; e++) {
                    double acc = 0;
                    for (int j = 0; j <= last; j++) {
                        if (in.patchValid()[r * n + j]) {
                            acc += w[j] * qkv[(r * n + j) * 3 * d + 2 * d + hh * hd + e];
                        }
                    }
                    out[oo + e] = (float) (acc / sum);
                }
            }
        });
    }

    /** Attention across the rows of each series at each patch; a known-future covariate reads only its peers. */
    private void groupAttention(float[] qkv, float[] out, Input in) {
        int rows = in.rows(), n = in.patches(), d = config.embedDim(), heads = config.numHeads(), hd = config.headDim();
        double scale = 1.0 / Math.sqrt(hd);
        IntStream.range(0, rows * heads).parallel().forEach(job -> {
            int a = job / heads, hh = job % heads, g = in.groupOf()[a];
            int start = in.groupStart()[g], end = in.groupStart()[g + 1];
            double[] w = new double[end - start];
            for (int s = 0; s < n; s++) {
                if (!in.patchValid()[a * n + s]) {
                    continue;
                }
                int qo = (a * n + s) * 3 * d + hh * hd;
                double max = Double.NEGATIVE_INFINITY;
                for (int m = start; m < end; m++) {
                    int b = in.groupRows()[m];
                    w[m - start] = Double.NaN;
                    if (!in.patchValid()[b * n + s] || in.future()[a] && !in.future()[b]) {
                        continue;
                    }
                    int ko = (b * n + s) * 3 * d + d + hh * hd;
                    double dot = 0;
                    for (int e = 0; e < hd; e++) {
                        dot += (double) qkv[qo + e] * qkv[ko + e];
                    }
                    w[m - start] = dot * scale;
                    max = Math.max(max, w[m - start]);
                }
                double sum = 0;
                for (int m = start; m < end; m++) {
                    if (!Double.isNaN(w[m - start])) {
                        w[m - start] = Math.exp(w[m - start] - max);
                        sum += w[m - start];
                    }
                }
                int oo = (a * n + s) * d + hh * hd;
                for (int e = 0; e < hd; e++) {
                    double acc = 0;
                    for (int m = start; m < end; m++) {
                        if (!Double.isNaN(w[m - start])) {
                            acc += w[m - start] * qkv[(in.groupRows()[m] * n + s) * 3 * d + 2 * d + hh * hd + e];
                        }
                    }
                    out[oo + e] = (float) (acc / sum);
                }
            }
        });
    }

    /** The decoder and the quantile head over {@code tokens} hidden vectors; {@code decoded[0]} gets the raw output. */
    private float[] head(float[] hidden, int tokens, float[][] decoded) {
        int q = config.quantiles(), p = config.patchSize();
        float[] raw = residualBlock(hidden, tokens, decoder);
        if (decoded != null) {
            decoded[0] = raw.clone();
        }
        float[] out = new float[raw.length];
        for (int i = 0; i < tokens * p; i++) {
            int o = i * q;
            float first = raw[o];
            out[o] = first;
            double cum = 0;
            for (int k = 1; k < q; k++) {
                cum += softplus(raw[o + k]);
                out[o + k] = first + (float) cum;
            }
        }
        return out;
    }

    /** {@return PyTorch's softplus (threshold 20), rounded to float} */
    public static float softplus(float x) {
        return x > 20f ? x : (float) Math.log1p(Math.exp(x));
    }
}
