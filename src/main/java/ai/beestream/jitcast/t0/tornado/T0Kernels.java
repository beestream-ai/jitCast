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

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * t0's own device kernels; its GEMMs, RMSNorms and SwiGLU are the shared ones
 * ({@link ai.beestream.jitcast.tornado.Kernels}). Activations are row-major {@code [rows · patches, …]};
 * {@code valid[t]} says whether patch {@code t} holds anything but padding, {@code future[r]} whether row
 * {@code r} is a known-future covariate.
 */
public final class T0Kernels {

    private T0Kernels() {}

    /** Work-group width of {@link #qkNormRope}: one thread per pair of q, then of k. */
    public static final int QK_THREADS = 128;

    /** Work-group width of {@link #timeAttention}. */
    public static final int ATTN_THREADS = 128;

    /** Queries per work-group of {@link #timeAttention}. */
    public static final int ATTN_QUERIES = 16;

    /** Keys staged in local memory at a time by {@link #timeAttention}. */
    public static final int ATTN_KEYS = 16;

    /** The widest attention head the kernels take. */
    public static final int HEAD_DIM = 128;

    /** Work-group width of {@link #groupAttention}: one thread per head dimension. */
    public static final int GROUP_THREADS = 128;

    /** Work-group width of {@link #biasRmsNorm}. */
    public static final int NORM_THREADS = 256;

    /** {@code x[t, c] += types[type[t], c]}: each patch's variate-type embedding. */
    public static void addType(KernelContext ctx, FloatArray x, FloatArray w, int typeOff, IntArray type, int tokens, int d) {
        int id = ctx.globalIdx;
        if (id < tokens * d) {
            int t = id / d;
            int c = id - t * d;
            x.set(id, x.get(id) + w.get(typeOff + type.get(t) * d + c));
        }
    }

    /**
     * In place on the {@code q|k|v} projections ({@code [tokens, 3·d]}): first the projection's bias when
     * {@code bias >= 0} (to q, k and v: the GEMM before it ran without one), then RMSNorm of each head's q and
     * k with their scales, then, when {@code rope} is 1, the rotary embedding on adjacent pairs at each patch's
     * position in the window with xPos scaling: q times {@code base^power}, k divided by it, where
     * {@code power = (position − ⌊(n − 1)/2⌋) / 512} and each coordinate's base is that of its pair
     * ({@code halves} 0: interleaved) or of its index modulo {@code hd/2} ({@code halves} 1). One work-group
     * of {@link #QK_THREADS} per (token, head): thread {@code i < hd/2} holds pair {@code i} of q, thread
     * {@code hd/2 + i} pair {@code i} of k.
     */
    public static void qkNormRope(KernelContext ctx, FloatArray qkv, FloatArray w, int bias, int qNorm, int kNorm, int invFreq,
            int logXpos, int rope, int halves, int rows, int n, int heads, int hd, int d, float eps) {
        if (ctx.groupIdx < rows * n * heads) {
            int head = ctx.groupIdx % heads;
            int tok = ctx.groupIdx / heads;
            int tid = ctx.localIdx;
            int half = hd / 2;
            float[] part = ctx.allocateFloatLocalArray(QK_THREADS);
            float[] sums = ctx.allocateFloatLocalArray(2);
            boolean active = tid < hd;
            int which = tid < half ? 0 : 1;
            int i = tid - which * half;
            int base = tok * 3 * d + which * d + head * hd;
            float a = 0f;
            float b = 0f;
            if (active) {
                a = qkv.get(base + 2 * i);
                b = qkv.get(base + 2 * i + 1);
                if (bias >= 0) {
                    int bo = bias + which * d + head * hd + 2 * i;
                    a += w.get(bo);
                    b += w.get(bo + 1);
                    int v = tok * 3 * d + 2 * d + head * hd + tid;
                    qkv.set(v, qkv.get(v) + w.get(bias + 2 * d + head * hd + tid));
                }
                part[tid] = a * a + b * b;
            }
            ctx.localBarrier();
            if (tid < 2) {
                float acc = 0f;
                for (int j = 0; j < half; j++) {
                    acc += part[tid * half + j];
                }
                sums[tid] = acc;
            }
            ctx.localBarrier();
            if (active) {
                float inv = 1f / TornadoMath.sqrt(sums[which] / hd + eps);
                int sOff = which == 0 ? qNorm : kNorm;
                float na = a * inv * w.get(sOff + 2 * i);
                float nb = b * inv * w.get(sOff + 2 * i + 1);
                if (rope == 1) {
                    int s = tok % n;
                    float ang = (float) s * w.get(invFreq + i);
                    float c = TornadoMath.cos(ang);
                    float sn = TornadoMath.sin(ang);
                    float power = ((float) s - (float) ((n - 1) / 2)) / 512f;
                    int ia = halves == 1 ? (2 * i) % half : i;
                    int ib = halves == 1 ? (2 * i + 1) % half : i;
                    float sa = TornadoMath.exp(power * w.get(logXpos + ia));
                    float sb = TornadoMath.exp(power * w.get(logXpos + ib));
                    if (which == 1) {
                        sa = 1f / sa;
                        sb = 1f / sb;
                    }
                    float ra = na * c * sa - nb * sn * sa;
                    float rb = nb * c * sb + na * sn * sb;
                    na = ra;
                    nb = rb;
                }
                qkv.set(base + 2 * i, na);
                qkv.set(base + 2 * i + 1, nb);
            }
        }
    }

    /**
     * Attention along each row's patches per (row, head): a valid query attends to the valid keys up to itself,
     * or to all of them in a known-future covariate's row; an all-padding query gets zeros. Flash-style: one
     * work-group of {@link #ATTN_THREADS} per (row, head, {@link #ATTN_QUERIES} queries) walks the keys in
     * blocks of {@link #ATTN_KEYS} with an online softmax. Each block's keys are staged transposed with a
     * padded stride, so a warp's score dot products read distinct banks, then its values in the same buffer;
     * the queries (padded likewise), scores and output accumulators also sit in local memory.
     */
    public static void timeAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray valid, IntArray future, int rows,
            int n, int heads, int hd, int d, float scale) {
        int tiles = (n + ATTN_QUERIES - 1) / ATTN_QUERIES;
        if (ctx.groupIdx < rows * heads * tiles) {
            int tile = ctx.groupIdx % tiles;
            int rh = ctx.groupIdx / tiles;
            int head = rh % heads;
            int r = rh / heads;
            int tid = ctx.localIdx;
            int qStride = HEAD_DIM + 1;
            int kStride = ATTN_KEYS + 1;
            float[] qs = ctx.allocateFloatLocalArray(ATTN_QUERIES * (HEAD_DIM + 1));
            float[] kv = ctx.allocateFloatLocalArray(HEAD_DIM * (ATTN_KEYS + 1));
            float[] sc = ctx.allocateFloatLocalArray(ATTN_QUERIES * ATTN_KEYS);
            float[] acc = ctx.allocateFloatLocalArray(ATTN_QUERIES * HEAD_DIM);
            float[] mx = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] den = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] alpha = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            boolean bidirectional = future.get(r) == 1;
            int q0 = tile * ATTN_QUERIES;
            int qEnd = TornadoMath.min(q0 + ATTN_QUERIES, n);
            int kEnd = bidirectional ? n : qEnd;
            int row0 = r * n;
            int qkvRow = 3 * d;
            for (int e = tid; e < ATTN_QUERIES * hd; e += ATTN_THREADS) {
                int qi = e / hd;
                int dd = e - qi * hd;
                int t = q0 + qi;
                qs[qi * qStride + dd] = t < n ? qkv.get((row0 + t) * qkvRow + head * hd + dd) * scale : 0f;
                acc[qi * HEAD_DIM + dd] = 0f;
            }
            if (tid < ATTN_QUERIES) {
                mx[tid] = -3.0e38f;
                den[tid] = 0f;
            }
            for (int k0 = 0; k0 < kEnd; k0 += ATTN_KEYS) {
                // keys, transposed: kv[dd · (ATTN_KEYS + 1) + kj]
                for (int e = tid; e < ATTN_KEYS * hd; e += ATTN_THREADS) {
                    int kj = e / hd;
                    int dd = e - kj * hd;
                    int key = k0 + kj;
                    kv[dd * kStride + kj] = key < n ? qkv.get((row0 + key) * qkvRow + d + head * hd + dd) : 0f;
                }
                ctx.localBarrier();
                for (int e = tid; e < ATTN_QUERIES * ATTN_KEYS; e += ATTN_THREADS) {
                    int qi = e / ATTN_KEYS;
                    int kj = e - qi * ATTN_KEYS;
                    int t = q0 + qi;
                    int key = k0 + kj;
                    float s = -3.0e38f;
                    if (t < n && key < n && valid.get(row0 + t) == 1 && valid.get(row0 + key) == 1 && (bidirectional || key <= t)) {
                        float dot = 0f;
                        for (int dd = 0; dd < hd; dd++) {
                            dot += qs[qi * qStride + dd] * kv[dd * kStride + kj];
                        }
                        s = dot;
                    }
                    sc[e] = s;
                }
                ctx.localBarrier();
                // the block's values replace its keys, while the first threads update the softmax
                for (int e = tid; e < ATTN_KEYS * hd; e += ATTN_THREADS) {
                    int kj = e / hd;
                    int dd = e - kj * hd;
                    int key = k0 + kj;
                    kv[kj * HEAD_DIM + dd] = key < n ? qkv.get((row0 + key) * qkvRow + 2 * d + head * hd + dd) : 0f;
                }
                if (tid < ATTN_QUERIES) {
                    float old = mx[tid];
                    float nm = old;
                    for (int kj = 0; kj < ATTN_KEYS; kj++) {
                        nm = TornadoMath.max(nm, sc[tid * ATTN_KEYS + kj]);
                    }
                    float sum = 0f;
                    for (int kj = 0; kj < ATTN_KEYS; kj++) {
                        float s = sc[tid * ATTN_KEYS + kj];
                        float pr = s > -1.0e38f ? TornadoMath.exp(s - nm) : 0f;
                        sc[tid * ATTN_KEYS + kj] = pr;
                        sum += pr;
                    }
                    float a = old > -1.0e38f ? TornadoMath.exp(old - nm) : 0f;
                    alpha[tid] = a;
                    den[tid] = den[tid] * a + sum;
                    mx[tid] = nm;
                }
                ctx.localBarrier();
                for (int e = tid; e < ATTN_QUERIES * hd; e += ATTN_THREADS) {
                    int qi = e / hd;
                    int dd = e - qi * hd;
                    float o = acc[qi * HEAD_DIM + dd] * alpha[qi];
                    for (int kj = 0; kj < ATTN_KEYS; kj++) {
                        o += sc[qi * ATTN_KEYS + kj] * kv[kj * HEAD_DIM + dd];
                    }
                    acc[qi * HEAD_DIM + dd] = o;
                }
                ctx.localBarrier();
            }
            for (int e = tid; e < ATTN_QUERIES * hd; e += ATTN_THREADS) {
                int qi = e / hd;
                int dd = e - qi * hd;
                int t = q0 + qi;
                if (t < n) {
                    float dn = den[qi];
                    out.set((row0 + t) * d + head * hd + dd, dn > 0f ? acc[qi * HEAD_DIM + dd] / dn : 0f);
                }
            }
        }
    }

    /**
     * Attention across the rows of each series at each patch, per (row, patch, head): a valid query attends to
     * the valid patches of its series' rows at the same position; a known-future covariate reads only
     * known-future covariates. One work-group of {@link #GROUP_THREADS}, one thread per head dimension, walks
     * the series' rows ({@code groupRows[groupStart[g] …]}) with an online softmax, each dot product reduced
     * in local memory.
     */
    public static void groupAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray valid, IntArray future,
            IntArray groupOf, IntArray groupStart, IntArray groupRows, int rows, int n, int heads, int hd, int d, float scale) {
        if (ctx.groupIdx < rows * n * heads) {
            int head = ctx.groupIdx % heads;
            int tok = ctx.groupIdx / heads;
            int a = tok / n;
            int s = tok - a * n;
            int tid = ctx.localIdx;
            float[] part = ctx.allocateFloatLocalArray(GROUP_THREADS);
            boolean active = tid < hd;
            float q = active ? qkv.get(tok * 3 * d + head * hd + tid) : 0f;
            float mx = -3.0e38f;
            float den = 0f;
            float acc = 0f;
            int g = groupOf.get(a);
            int start = groupStart.get(g);
            int end = groupStart.get(g + 1);
            boolean query = valid.get(tok) == 1;
            boolean fa = future.get(a) == 1;
            for (int m = start; m < end; m++) {
                int b = groupRows.get(m);
                int kt = b * n + s;
                boolean allowed = query && valid.get(kt) == 1 && !(fa && future.get(b) == 0);
                if (allowed) {
                    part[tid] = active ? q * qkv.get(kt * 3 * d + d + head * hd + tid) : 0f;
                    ctx.localBarrier();
                    for (int st = GROUP_THREADS / 2; st > 0; st >>= 1) {
                        if (tid < st) {
                            part[tid] += part[tid + st];
                        }
                        ctx.localBarrier();
                    }
                    float score = part[0] * scale;
                    ctx.localBarrier();
                    float nm = TornadoMath.max(mx, score);
                    float al = mx > -1.0e38f ? TornadoMath.exp(mx - nm) : 0f;
                    float pr = TornadoMath.exp(score - nm);
                    den = den * al + pr;
                    float v = active ? qkv.get(kt * 3 * d + 2 * d + head * hd + tid) : 0f;
                    acc = acc * al + pr * v;
                    mx = nm;
                }
            }
            if (active) {
                out.set(tok * d + head * hd + tid, den > 0f ? acc / den : 0f);
            }
        }
    }

    /**
     * {@code x[r] += b} when {@code bOff >= 0} (the bias of the GEMM that accumulated into the residual stream
     * without one), then {@code y[r] = RMSNorm(x[r])·w}: one work-group of {@link #NORM_THREADS} per row.
     */
    public static void biasRmsNorm(KernelContext ctx, FloatArray x, FloatArray y, FloatArray w, int bOff, int wOff, int d,
            float eps, int rows) {
        if (ctx.groupIdx < rows) {
            int row = ctx.groupIdx;
            int tid = ctx.localIdx;
            float[] part = ctx.allocateFloatLocalArray(NORM_THREADS);
            int base = row * d;
            float ss = 0f;
            for (int i = tid; i < d; i += NORM_THREADS) {
                float v = x.get(base + i);
                if (bOff >= 0) {
                    v += w.get(bOff + i);
                    x.set(base + i, v);
                }
                ss += v * v;
            }
            part[tid] = ss;
            ctx.localBarrier();
            for (int s = NORM_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    part[tid] += part[tid + s];
                }
                ctx.localBarrier();
            }
            float inv = 1f / TornadoMath.sqrt(part[0] / d + eps);
            for (int i = tid; i < d; i += NORM_THREADS) {
                y.set(base + i, x.get(base + i) * inv * w.get(wOff + i));
            }
        }
    }

    /**
     * {@code out[r, j] = SiLU(g + b[j]) · (u + b[f + j])} on the gate|value projection {@code [n, 2f]} that
     * its GEMM wrote without the bias.
     */
    public static void biasSwiGlu(KernelContext ctx, FloatArray big, FloatArray out, FloatArray w, int bOff, int n, int ffn) {
        int id = ctx.globalIdx;
        if (id < n * ffn) {
            int r = id / ffn;
            int j = id - r * ffn;
            float g = big.get(r * 2 * ffn + j) + w.get(bOff + j);
            float u = big.get(r * 2 * ffn + ffn + j) + w.get(bOff + ffn + j);
            out.set(id, g / (1f + TornadoMath.exp(-g)) * u);
        }
    }

    /** {@code dst[r, k] = src[r, from + k]} for {@code count} patches of each of the {@code rows} rows. */
    public static void gatherPatches(KernelContext ctx, FloatArray src, FloatArray dst, int rows, int n, int from, int count, int d) {
        int id = ctx.globalIdx;
        if (id < rows * count * d) {
            int rk = id / d;
            int c = id - rk * d;
            int r = rk / count;
            int k = rk - r * count;
            dst.set(id, src.get((r * n + from + k) * d + c));
        }
    }

    /**
     * The quantile head on {@code positions} vectors of {@code q} raw values: the first value, then the first
     * plus the running sum of the softplus of the others (PyTorch's threshold of 20).
     */
    public static void quantileHead(KernelContext ctx, FloatArray raw, FloatArray out, int positions, int q) {
        int i = ctx.globalIdx;
        if (i < positions) {
            int o = i * q;
            float first = raw.get(o);
            out.set(o, first);
            float cum = 0f;
            for (int k = 1; k < q; k++) {
                float v = raw.get(o + k);
                float sp = v > 20f ? v : TornadoMath.log(1f + TornadoMath.exp(v));
                cum += sp;
                out.set(o + k, first + cum);
            }
        }
    }
}
