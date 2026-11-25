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

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Toto 2.0's own device kernels; its GEMMs, RMSNorms and SwiGLU are the shared ones
 * ({@link ai.beestream.jitcast.tornado.Kernels}). Activations are row-major {@code [rows · patches, …]};
 * {@code valid[r · patches + t]} is 1 when patch {@code t} of row {@code r} takes part in attention with the
 * other valid patches, 0 when it only sees the other invalid ones.
 */
public final class TotoKernels {

    private TotoKernels() {}

    /** Work-group width of {@link #timeAttention}. */
    public static final int ATTN_THREADS = 128;
    /** Queries per work-group of {@link #timeAttention}. */
    public static final int ATTN_QUERIES = 16;
    /** Keys staged in local memory at a time by {@link #timeAttention}. */
    public static final int ATTN_KEYS = 16;
    /** The widest attention head the kernels take. */
    public static final int HEAD_DIM = 64;
    /** Work-group width of {@link #variateAttention}: one thread per head dimension. */
    public static final int VAR_THREADS = 64;
    /** Values per rotated pair in the rotation table: cos, sin, the query's and the key's xPos scale. */
    public static final int ROT = 4;

    /**
     * In place on the {@code q|k|v} projections ({@code [tokens, 3·d]}): the rotary embedding with xPos
     * scaling on the first {@code 2·pairs} dimensions of each head of q and k (interleaved pairs). The table
     * {@code rot} holds, per patch and pair, {@code cos, sin, xq, xk}. One thread per (token, head, pair, q|k).
     */
    public static void rope(KernelContext ctx, FloatArray qkv, FloatArray rot, int rows, int n, int heads, int hd, int d, int pairs) {
        int id = ctx.globalIdx;
        if (id < rows * n * heads * pairs * 2) {
            int k = id % pairs;
            int rest = id / pairs;
            int which = rest % 2;
            rest = rest / 2;
            int head = rest % heads;
            int tok = rest / heads;
            int t = tok % n;
            int r0 = (t * pairs + k) * ROT;
            float c = rot.get(r0);
            float s = rot.get(r0 + 1);
            float x = rot.get(r0 + 2 + which);
            int base = tok * 3 * d + which * d + head * hd + 2 * k;
            float a = qkv.get(base);
            float b = qkv.get(base + 1);
            qkv.set(base, (a * c - b * s) * x);
            qkv.set(base + 1, (b * c + a * s) * x);
        }
    }

    /**
     * In place on the {@code q|k|v} projections of a GEMM that ran without its bias: adds the bias, then, when
     * {@code rope} is 1, applies {@link #rope}'s rotation and xPos scaling to the first {@code 2·pairs}
     * dimensions of each head of q and k. One thread per pair of adjacent columns.
     */
    public static void biasRope(KernelContext ctx, FloatArray qkv, FloatArray w, int bOff, FloatArray rot, int rows, int n, int heads,
            int hd, int d, int pairs, int rope) {
        int half = 3 * d / 2;
        int id = ctx.globalIdx;
        if (id < rows * n * half) {
            int j = id % half;
            int tok = id / half;
            int c = 2 * j;
            int base = tok * 3 * d + c;
            float a = qkv.get(base) + w.get(bOff + c);
            float b = qkv.get(base + 1) + w.get(bOff + c + 1);
            int which = c / d;
            int hc = (c - which * d) % hd;
            if (rope == 1 && which < 2 && hc < 2 * pairs) {
                int t = tok % n;
                int r0 = (t * pairs + hc / 2) * ROT;
                float cs = rot.get(r0);
                float sn = rot.get(r0 + 1);
                float x = rot.get(r0 + 2 + which);
                float na = (a * cs - b * sn) * x;
                b = (b * cs + a * sn) * x;
                a = na;
            }
            qkv.set(base, a);
            qkv.set(base + 1, b);
        }
    }

    /**
     * Causal attention per (row, head) over time, a query seeing the earlier keys of the same validity, with
     * the queries already scaled. Flash-style: one work-group of {@link #ATTN_THREADS} per (row, head,
     * {@link #ATTN_QUERIES} queries) walks the keys in blocks of {@link #ATTN_KEYS} with an online softmax.
     * Each block's keys are staged transposed with a padded stride, so that the score dot products are free
     * of bank conflicts, and its values then reuse the same local buffer.
     */
    public static void timeAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray valid, int rows, int n, int heads,
            int hd, int d) {
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
            int q0 = tile * ATTN_QUERIES;
            int qEnd = TornadoMath.min(q0 + ATTN_QUERIES, n);
            int row0 = r * n;
            int qkvRow = 3 * d;
            for (int e = tid; e < ATTN_QUERIES * hd; e += ATTN_THREADS) {
                int qi = e / hd;
                int dd = e - qi * hd;
                int t = q0 + qi;
                qs[qi * qStride + dd] = t < n ? qkv.get((row0 + t) * qkvRow + head * hd + dd) : 0f;
                acc[qi * HEAD_DIM + dd] = 0f;
            }
            if (tid < ATTN_QUERIES) {
                mx[tid] = -3.0e38f;
                den[tid] = 0f;
            }
            for (int k0 = 0; k0 < qEnd; k0 += ATTN_KEYS) {
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
                    if (t < n && key <= t && valid.get(row0 + key) == valid.get(row0 + t)) {
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
                        float sv = sc[tid * ATTN_KEYS + kj];
                        float pr = sv > -1.0e38f ? TornadoMath.exp(sv - nm) : 0f;
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
                    out.set((row0 + t) * d + head * hd + dd, acc[qi * HEAD_DIM + dd] / den[qi]);
                }
            }
        }
    }

    /**
     * Attention across the rows of one series at each patch, without positions: row {@code r} attends to rows
     * {@code first[r]} to {@code first[r] + count[r] − 1} whose patch has its validity. One work-group of
     * {@link #VAR_THREADS} per (row, patch, head), thread {@code i} owning output dimension {@code i}.
     */
    public static void variateAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray valid, IntArray first,
            IntArray count, int rows, int n, int heads, int hd, int d) {
        if (ctx.groupIdx < rows * n * heads) {
            int head = ctx.groupIdx % heads;
            int tok = ctx.groupIdx / heads;
            int r = tok / n;
            int t = tok - r * n;
            int tid = ctx.localIdx;
            float[] part = ctx.allocateFloatLocalArray(VAR_THREADS);
            int v0 = valid.get(tok);
            int f = first.get(r);
            int m = count.get(r);
            boolean active = tid < hd;
            float q = active ? qkv.get(tok * 3 * d + head * hd + tid) : 0f;
            float mxv = -3.0e38f;
            float den = 0f;
            float acc = 0f;
            for (int b = 0; b < m; b++) {
                int other = (f + b) * n + t;
                int ok = valid.get(other) == v0 ? 1 : 0;
                part[tid] = active && ok == 1 ? q * qkv.get(other * 3 * d + d + head * hd + tid) : 0f;
                ctx.localBarrier();
                for (int s = VAR_THREADS / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        part[tid] += part[tid + s];
                    }
                    ctx.localBarrier();
                }
                float score = part[0];
                ctx.localBarrier();
                if (ok == 1) {
                    float nm = TornadoMath.max(mxv, score);
                    float a = mxv > -1.0e38f ? TornadoMath.exp(mxv - nm) : 0f;
                    float pr = TornadoMath.exp(score - nm);
                    float vv = active ? qkv.get(other * 3 * d + 2 * d + head * hd + tid) : 0f;
                    acc = acc * a + pr * vv;
                    den = den * a + pr;
                    mxv = nm;
                }
            }
            if (active) {
                out.set(tok * d + head * hd + tid, acc / den);
            }
        }
    }

    /** {@code dst[r, f] = x[r, n − k − 1 + f]}: the patches whose outputs predict the {@code k} forecast patches. */
    public static void gather(KernelContext ctx, FloatArray x, FloatArray dst, int rows, int n, int k, int d) {
        int id = ctx.globalIdx;
        if (id < rows * k * d) {
            int i = id % d;
            int rf = id / d;
            int f = rf % k;
            int r = rf / k;
            dst.set(id, x.get((r * n + n - k - 1 + f) * d + i));
        }
    }

    /**
     * {@code x = x · keep + branch + b}: {@link #mix} with the bias of a branch whose GEMM ran without it,
     * {@code b} at {@code bOff} in {@code w}, per column of {@code d}.
     */
    public static void mixBias(KernelContext ctx, FloatArray x, FloatArray branch, float keep, FloatArray w, int bOff, int d, int count) {
        int i = ctx.globalIdx;
        if (i < count) {
            x.set(i, x.get(i) * keep + branch.get(i) + w.get(bOff + i % d));
        }
    }

    /** {@code x = x · keep + branch}: the residual stream around a branch whose weight is folded in. */
    public static void mix(KernelContext ctx, FloatArray x, FloatArray branch, float keep, int count) {
        int i = ctx.globalIdx;
        if (i < count) {
            x.set(i, x.get(i) * keep + branch.get(i));
        }
    }
}
