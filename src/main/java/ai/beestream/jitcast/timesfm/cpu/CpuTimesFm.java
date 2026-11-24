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
package ai.beestream.jitcast.timesfm.cpu;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import ai.beestream.jitcast.cpu.Ops;
import ai.beestream.jitcast.timesfm.PatchForward;
import ai.beestream.jitcast.timesfm.TimesFmConfig;
import ai.beestream.jitcast.timesfm.TimesFmWeights;

/**
 * The FP32 reference of TimesFM 2.5's forward pass ({@code TimesFM_2p5_200M_torch_module.forward}), the
 * oracle the device kernels are tested against. Each layer: RMSNorm, fused q|k|v, RoPE on q and k
 * (positions counted from each row's first observed patch), per-head RMSNorm of q and k, a learned
 * per-dimension query scale in place of {@code 1/√d}, causal attention over the observed patches, the
 * output projection and its RMSNorm plus the residual, then a SiLU feed-forward layer between two
 * RMSNorms plus the residual.
 */
public final class CpuTimesFm implements PatchForward {

    /** The query scale's constant: {@code 1/ln 2}, as the reference. */
    private static final double LOG2E = 1.442695041;

    /** RoPE's longest timescale. */
    private static final double MAX_TIMESCALE = 10000.0;

    private record Block(float[] hw, float[] hb, float[] ow, float[] ob, float[] rw, float[] rb, int in, int hidden, int out) {}

    private record Layer(float[] preAttn, float[] qkv, float[] qNorm, float[] kNorm, float[] queryScale, float[] out,
            float[] postAttn, float[] preFf, float[] ff0, float[] ff1, float[] postFf) {}

    /**
     * Every stage of one forward pass, for the golden tests.
     *
     * @param tokens the tokenizer's output, {@code [rows · patches · D]}
     * @param layers each layer's output
     * @param point the point head of every patch, {@code [rows · patches · horizonLength · outputs]}
     * @param heads the heads of each row's last patch
     */
    public record Stages(float[] tokens, List<float[]> layers, float[] point, Heads heads) {}

    private final TimesFmConfig config;
    private final Block tokenizer;
    private final List<Layer> layers;
    private final Block pointHead;
    private final Block quantileHead;
    private final double[] invTimescale;

    /**
     * Copies the weights it reads to the heap.
     *
     * @param weights the checkpoint
     */
    public CpuTimesFm(TimesFmWeights weights) {
        this.config = weights.config();
        this.tokenizer = block(weights.tokenizer());
        List<Layer> ls = new ArrayList<>();
        int hd = config.headDim();
        for (TimesFmWeights.Layer l : weights.layers()) {
            float[] pds = l.perDimScale().toFloatArray();
            float[] scale = new float[hd];
            for (int i = 0; i < hd; i++) {
                scale[i] = (float) (LOG2E / Math.sqrt(hd) * softplus(pds[i]));
            }
            ls.add(new Layer(l.preAttnNorm().toFloatArray(), l.qkv().weight().toFloatArray(), l.queryNorm().toFloatArray(),
                    l.keyNorm().toFloatArray(), scale, l.out().weight().toFloatArray(), l.postAttnNorm().toFloatArray(),
                    l.preFfNorm().toFloatArray(), l.ff0().weight().toFloatArray(), l.ff1().weight().toFloatArray(),
                    l.postFfNorm().toFloatArray()));
        }
        this.layers = List.copyOf(ls);
        this.pointHead = block(weights.pointHead());
        this.quantileHead = block(weights.quantileHead());
        int half = hd / 2;
        this.invTimescale = new double[half];
        for (int i = 0; i < half; i++) {
            invTimescale[i] = 1.0 / Math.pow(MAX_TIMESCALE, 2.0 * i / hd);
        }
    }

    private static double softplus(double x) {
        return x > 20 ? x : Math.log1p(Math.exp(x));
    }

    private static Block block(TimesFmWeights.ResidualBlock b) {
        return new Block(b.hidden().weight().toFloatArray(), b.hidden().bias() == null ? null : b.hidden().bias().toFloatArray(),
                b.output().weight().toFloatArray(), b.output().bias() == null ? null : b.output().bias().toFloatArray(),
                b.residual().weight().toFloatArray(), b.residual().bias() == null ? null : b.residual().bias().toFloatArray(),
                b.hidden().in(), b.hidden().out(), b.output().out());
    }

    private static float[] residualBlock(float[] x, int rows, Block b) {
        float[] h = Ops.linear(x, rows, b.hw, b.hb, b.in, b.hidden);
        Ops.siluInPlace(h);
        float[] y = Ops.linear(h, rows, b.ow, b.ob, b.hidden, b.out);
        Ops.addInPlace(y, Ops.linear(x, rows, b.rw, b.rb, b.in, b.out));
        return y;
    }

    @Override
    public Heads forward(Batch batch) {
        return run(batch, null);
    }

    /**
     * {@return every stage of one forward pass}
     *
     * @param batch the inputs
     */
    public Stages stages(Batch batch) {
        List<float[]> out = new ArrayList<>();
        Heads heads = run(batch, out);
        float[] tokens = out.remove(0);
        float[] point = out.remove(out.size() - 1);
        return new Stages(tokens, out, point, heads);
    }

    /** The forward pass; with {@code stages}, also the tokens, each layer and the point head of every patch. */
    private Heads run(Batch b, List<float[]> stages) {
        int rows = b.rows(), n = b.patches(), p = config.patchLength(), d = config.hiddenSize();
        int tokens = rows * n;
        float[] in = new float[tokens * 2 * p];
        int[] masked = new int[rows];
        for (int t = 0; t < tokens; t++) {
            System.arraycopy(b.values(), t * p, in, t * 2 * p, p);
            System.arraycopy(b.masks(), t * p, in, t * 2 * p + p, p);
            if (b.masks()[t * p + p - 1] > 0.5f) {
                masked[t / n]++;
            }
        }
        float[] x = residualBlock(in, tokens, tokenizer);
        if (stages != null) {
            stages.add(x.clone());
        }
        for (Layer l : layers) {
            x = layer(x, rows, n, masked, l);
            if (stages != null) {
                stages.add(x.clone());
            }
        }
        if (stages != null) {
            stages.add(residualBlock(x, tokens, pointHead));
        }
        float[] last = new float[rows * d];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(x, ((r + 1) * n - 1) * d, last, r * d, d);
        }
        return new Heads(residualBlock(last, rows, pointHead), b.spread() ? residualBlock(last, rows, quantileHead) : null);
    }

    private float[] layer(float[] x, int rows, int n, int[] masked, Layer l) {
        int d = config.hiddenSize(), tokens = rows * n;
        float eps = config.rmsNormEps();
        float[] h = Ops.rmsNorm(x, tokens, d, l.preAttn, eps);
        float[] qkv = Ops.linear(h, tokens, l.qkv, null, d, 3 * d);
        float[] attn = attention(qkv, rows, n, masked, l);
        float[] a = Ops.rmsNorm(Ops.linear(attn, tokens, l.out, null, d, d), tokens, d, l.postAttn, eps);
        Ops.addInPlace(a, x);
        float[] f = Ops.linear(Ops.rmsNorm(a, tokens, d, l.preFf, eps), tokens, l.ff0, null, d, config.intermediateSize());
        Ops.siluInPlace(f);
        float[] y = Ops.rmsNorm(Ops.linear(f, tokens, l.ff1, null, config.intermediateSize(), d), tokens, d, l.postFf, eps);
        Ops.addInPlace(y, a);
        return y;
    }

    /** Causal attention per row and head; a patch whose row has not started yet (padding) gets zeros. */
    private float[] attention(float[] qkv, int rows, int n, int[] masked, Layer l) {
        int d = config.hiddenSize(), heads = config.numHeads(), hd = config.headDim();
        float eps = config.rmsNormEps();
        float[] out = new float[rows * n * d];
        IntStream.range(0, rows * heads).parallel().forEach(rh -> {
            int r = rh / heads, head = rh % heads;
            float[][] q = new float[n][], k = new float[n][], v = new float[n][];
            for (int t = 0; t < n; t++) {
                int base = (r * n + t) * 3 * d + head * hd;
                double pos = t - masked[r];
                q[t] = prepare(qkv, base, pos, l.qNorm, eps);
                k[t] = prepare(qkv, base + d, pos, l.kNorm, eps);
                v[t] = new float[hd];
                System.arraycopy(qkv, base + 2 * d, v[t], 0, hd);
                for (int i = 0; i < hd; i++) {
                    q[t][i] *= l.queryScale[i];
                }
            }
            double[] s = new double[n];
            for (int t = masked[r]; t < n; t++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int j = masked[r]; j <= t; j++) {
                    double acc = 0;
                    for (int i = 0; i < hd; i++) {
                        acc += (double) q[t][i] * k[j][i];
                    }
                    s[j] = acc;
                    max = Math.max(max, acc);
                }
                double sum = 0;
                for (int j = masked[r]; j <= t; j++) {
                    s[j] = Math.exp(s[j] - max);
                    sum += s[j];
                }
                int o = (r * n + t) * d + head * hd;
                for (int i = 0; i < hd; i++) {
                    double acc = 0;
                    for (int j = masked[r]; j <= t; j++) {
                        acc += s[j] * v[j][i];
                    }
                    out[o + i] = (float) (acc / sum);
                }
            }
        });
        return out;
    }

    /** RoPE at {@code pos} (rotating the first half against the second), then RMSNorm with {@code scale}. */
    private float[] prepare(float[] qkv, int base, double pos, float[] scale, float eps) {
        int hd = config.headDim(), half = hd / 2;
        float[] y = new float[hd];
        for (int i = 0; i < half; i++) {
            double angle = pos * invTimescale[i];
            double c = Math.cos(angle), s = Math.sin(angle);
            double a = qkv[base + i], b = qkv[base + half + i];
            y[i] = (float) (a * c - b * s);
            y[half + i] = (float) (b * c + a * s);
        }
        double ss = 0;
        for (float f : y) {
            ss += (double) f * f;
        }
        float inv = (float) (1.0 / Math.sqrt(ss / hd + eps));
        for (int i = 0; i < hd; i++) {
            y[i] = y[i] * inv * scale[i];
        }
        return y;
    }
}
