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
package ai.beestream.jitcast.toto.cpu;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import ai.beestream.jitcast.cpu.Ops;
import ai.beestream.jitcast.toto.TotoConfig;
import ai.beestream.jitcast.toto.TotoForward;
import ai.beestream.jitcast.toto.TotoWeights;

/**
 * The reference forward pass of Toto 2.0 on the CPU, in float32 storage with double accumulation. Follows
 * Datadog's {@code toto2} module: a residual-MLP patch embedding, pre-norm layers whose residual stream is
 * mixed as {@code (τ·branch + x) / √(1 + τ²)}, time attention with a rotary embedding and xPos scaling on
 * the first half of every head, variate attention without positions, a SwiGLU feed-forward and a
 * residual-MLP quantile head, all with the unit-scaling constants applied.
 */
public final class CpuToto implements TotoForward {

    /** xPos' scale base: a position offset of this many patches scales by one factor of the per-pair base. */
    private static final double XPOS_SCALE_BASE = 256;
    private static final double ROPE_BASE = 10000;

    private record Lin(float[] w, float[] b, int in, int out, float scale) {}

    private record Mlp(Lin l1, Lin l2, Lin skip) {}

    private record Layer(Lin inProj, float[] queryScale, Lin outProj, double attnTau, Lin fc1, Lin fc2, double mlpTau) {}

    /**
     * Every stage of one forward pass, each {@code [rows, patches, d]} except the head.
     *
     * @param tokens the patch embedding
     * @param layers each layer's output
     * @param outNorm the final norm's output
     * @param head the forward's result
     */
    public record Stages(float[] tokens, List<float[]> layers, float[] outNorm, float[] head) {}

    private final TotoConfig config;
    private final Mlp patchProj;
    private final List<Layer> layers;
    private final Mlp head;
    private final double[] theta;
    private final double[] xposBase;

    /**
     * A forward pass over a checkpoint's weights, copied to the heap.
     *
     * @param weights the checkpoint
     */
    public CpuToto(TotoWeights weights) {
        this.config = weights.config();
        this.patchProj = mlp(weights.patchProj(), false);
        List<Layer> ls = new ArrayList<>();
        int hd = config.headDim();
        for (TotoWeights.Layer l : weights.layers()) {
            float[] pds = l.perDimScale().toFloatArray();
            float[] scale = new float[hd];
            for (int i = 0; i < hd; i++) {
                scale[i] = (float) (softplus(pds[i]) / Math.log(2.0));
            }
            ls.add(new Layer(lin(l.inProj(), false), scale, lin(l.outProj(), false), l.attnTau(), lin(l.fc1(), false),
                    lin(l.fc2(), false), l.mlpTau()));
        }
        this.layers = List.copyOf(ls);
        this.head = mlp(weights.head(), true);
        int w = config.ropeWidth(), pairs = w / 2;
        this.theta = new double[pairs];
        this.xposBase = new double[pairs];
        for (int k = 0; k < pairs; k++) {
            theta[k] = 1.0 / Math.pow(ROPE_BASE, (2.0 * k) / w);
            xposBase[k] = (2.0 * k + 0.4 * w) / (1.4 * w);
        }
    }

    private static double softplus(double x) {
        return x > 20 ? x : Math.log1p(Math.exp(x));
    }

    private static Lin lin(TotoWeights.Linear l, boolean readout) {
        int in = l.in();
        return new Lin(l.weight().toFloatArray(), l.bias() == null ? null : l.bias().toFloatArray(), in, l.out(),
                (float) (readout ? 1.0 / in : 1.0 / Math.sqrt(in)));
    }

    private static Mlp mlp(TotoWeights.ResidualMlp m, boolean output) {
        return new Mlp(lin(m.linear1(), false), lin(m.linear2(), output), lin(m.skip(), output));
    }

    private static float[] apply(float[] x, int rows, Lin l) {
        float[] y = Ops.linear(x, rows, l.w, l.b, l.in, l.out);
        for (int i = 0; i < y.length; i++) {
            y[i] *= l.scale;
        }
        return y;
    }

    private static float[] residualMlp(float[] x, int rows, Mlp m) {
        float[] h = apply(x, rows, m.l1);
        for (int i = 0; i < h.length; i++) {
            h[i] = (float) (SILU_SCALE * Ops.silu(h[i]));
        }
        float[] y = apply(h, rows, m.l2);
        float[] s = apply(x, rows, m.skip);
        for (int i = 0; i < y.length; i++) {
            y[i] = (float) ((y[i] + (double) s[i]) * RESIDUAL_MLP_SCALE);
        }
        return y;
    }

    @Override
    public float[] forward(Input in) {
        return run(in, null).head;
    }

    /**
     * Runs the forward pass and keeps every stage.
     *
     * @param in the batch
     * @return the stages
     */
    public Stages stages(Input in) {
        return run(in, new ArrayList<>());
    }

    private Stages run(Input in, List<float[]> layerOut) {
        int rows = in.rows(), n = in.patches(), p = config.patchSize(), d = config.dModel(), tokens = rows * n;
        float[] x0 = new float[tokens * 2 * p];
        for (int t = 0; t < tokens; t++) {
            System.arraycopy(in.values(), t * p, x0, t * 2 * p, p);
            System.arraycopy(in.missing(), t * p, x0, t * 2 * p + p, p);
        }
        float[] x = residualMlp(x0, tokens, patchProj);
        float[] tok = layerOut == null ? null : x.clone();
        for (int i = 0; i < layers.size(); i++) {
            x = layer(x, in, layers.get(i), config.variateLayer(i));
            if (layerOut != null) {
                layerOut.add(x.clone());
            }
        }
        float[] normed = Ops.rmsNorm(x, tokens, d, null, config.normEps());
        int k = in.forecast(), q = config.quantiles();
        float[] last = new float[rows * k * d];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(normed, (r * n + n - k - 1) * d, last, r * k * d, k * d);
        }
        float[] out = residualMlp(last, rows * k, head);
        assert out.length == rows * k * p * q;
        return new Stages(tok, layerOut, layerOut == null ? null : normed, out);
    }

    private float[] layer(float[] x, Input in, Layer l, boolean variate) {
        int d = config.dModel(), tokens = in.rows() * in.patches();
        float eps = config.normEps();
        float[] qkv = apply(Ops.rmsNorm(x, tokens, d, null, eps), tokens, l.inProj);
        float[] attn = variate ? variateAttention(qkv, in, l) : timeAttention(qkv, in, l);
        float[] a = apply(attn, tokens, l.outProj);
        mix(x, a, l.attnTau);
        float[] f = apply(Ops.rmsNorm(x, tokens, d, null, eps), tokens, l.fc1);
        int ff = config.dFf();
        float[] g = new float[tokens * ff];
        for (int t = 0; t < tokens; t++) {
            for (int i = 0; i < ff; i++) {
                g[t * ff + i] = f[t * 2 * ff + i] * Ops.silu(f[t * 2 * ff + ff + i]);
            }
        }
        float[] y = apply(g, tokens, l.fc2);
        mix(x, y, l.mlpTau);
        return x;
    }

    /** {@code x ← (τ·branch + x) / √(1 + τ²)}, in place. */
    private static void mix(float[] x, float[] branch, double tau) {
        double inv = 1.0 / Math.sqrt(1 + tau * tau);
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) ((tau * branch[i] + x[i]) * inv);
        }
    }

    private float[] timeAttention(float[] qkv, Input in, Layer l) {
        int rows = in.rows(), n = in.patches(), d = config.dModel(), heads = config.numHeads(), hd = config.headDim();
        double scale = TotoForward.attentionScale(config);
        int center = n / 2;
        float[] out = new float[rows * n * d];
        IntStream.range(0, rows * heads).parallel().forEach(rh -> {
            int r = rh / heads, head = rh % heads;
            float[][] q = new float[n][], k = new float[n][], v = new float[n][];
            for (int t = 0; t < n; t++) {
                int base = (r * n + t) * 3 * d + head * hd;
                q[t] = new float[hd];
                for (int i = 0; i < hd; i++) {
                    q[t][i] = qkv[base + i] * l.queryScale[i];
                }
                rotate(q[t], t, center, 1);
                k[t] = new float[hd];
                System.arraycopy(qkv, base + d, k[t], 0, hd);
                rotate(k[t], t, center, -1);
                v[t] = new float[hd];
                System.arraycopy(qkv, base + 2 * d, v[t], 0, hd);
            }
            double[] s = new double[n];
            for (int t = 0; t < n; t++) {
                boolean vt = in.valid()[r * n + t];
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j <= t; j++) {
                    if (in.valid()[r * n + j] != vt) {
                        s[j] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    double acc = 0;
                    for (int i = 0; i < hd; i++) {
                        acc += (double) q[t][i] * k[j][i];
                    }
                    s[j] = acc * scale;
                    max = Math.max(max, s[j]);
                }
                softmaxInto(out, (r * n + t) * d + head * hd, s, t + 1, max, v, hd);
            }
        });
        return out;
    }

    private float[] variateAttention(float[] qkv, Input in, Layer l) {
        int n = in.patches(), d = config.dModel(), heads = config.numHeads(), hd = config.headDim();
        double scale = TotoForward.attentionScale(config);
        float[] out = new float[in.rows() * n * d];
        IntStream.range(0, in.series() * n * heads).parallel().forEach(job -> {
            int sIdx = job / (n * heads), t = (job / heads) % n, head = job % heads;
            int r0 = in.seriesStart()[sIdx], m = in.seriesStart()[sIdx + 1] - r0;
            float[][] k = new float[m][], v = new float[m][];
            for (int a = 0; a < m; a++) {
                int base = ((r0 + a) * n + t) * 3 * d + head * hd;
                k[a] = new float[hd];
                System.arraycopy(qkv, base + d, k[a], 0, hd);
                v[a] = new float[hd];
                System.arraycopy(qkv, base + 2 * d, v[a], 0, hd);
            }
            double[] s = new double[m];
            for (int a = 0; a < m; a++) {
                int base = ((r0 + a) * n + t) * 3 * d + head * hd;
                boolean va = in.valid()[(r0 + a) * n + t];
                double max = Double.NEGATIVE_INFINITY;
                for (int b = 0; b < m; b++) {
                    if (in.valid()[(r0 + b) * n + t] != va) {
                        s[b] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    double acc = 0;
                    for (int i = 0; i < hd; i++) {
                        acc += (double) (qkv[base + i] * l.queryScale[i]) * k[b][i];
                    }
                    s[b] = acc * scale;
                    max = Math.max(max, s[b]);
                }
                softmaxInto(out, ((r0 + a) * n + t) * d + head * hd, s, m, max, v, hd);
            }
        });
        return out;
    }

    private static void softmaxInto(float[] out, int o, double[] s, int count, double max, float[][] v, int hd) {
        double sum = 0;
        for (int j = 0; j < count; j++) {
            s[j] = s[j] == Double.NEGATIVE_INFINITY ? 0 : Math.exp(s[j] - max);
            sum += s[j];
        }
        for (int i = 0; i < hd; i++) {
            double acc = 0;
            for (int j = 0; j < count; j++) {
                acc += s[j] * v[j][i];
            }
            out[o + i] = (float) (acc / sum);
        }
    }

    /**
     * The rotary embedding on the first half of a head (interleaved pairs), with xPos scaling: a query
     * ({@code sign = 1}) is scaled by {@code base_k^((pos − center)/256)}, a key ({@code sign = −1}) by its
     * inverse, so that their product depends on the distance only.
     */
    private void rotate(float[] x, int pos, int center, int sign) {
        double rel = (double) (pos - center) / XPOS_SCALE_BASE;
        for (int k = 0; k < theta.length; k++) {
            double angle = pos * theta[k];
            double c = Math.cos(angle), s = Math.sin(angle);
            double a = x[2 * k], b = x[2 * k + 1];
            double sc = Math.pow(xposBase[k], sign * rel);
            x[2 * k] = (float) ((a * c - b * s) * sc);
            x[2 * k + 1] = (float) ((b * c + a * s) * sc);
        }
    }
}
