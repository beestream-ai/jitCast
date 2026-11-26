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

import ai.beestream.jitcast.tornado.Kernels;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
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

    /** Warps per work-group of {@link #timeAttentionMma}; its threads, queries (16 per warp), keys per block and head width. */
    public static final int MMA_WARPS = 2;
    public static final int MMA_THREADS = 32 * MMA_WARPS;
    public static final int MMA_QUERIES = 16 * MMA_WARPS;
    public static final int MMA_KEYS = 32;
    public static final int MMA_HEAD_DIM = 128;
    /** Shared-memory row pitches, in FP16 pairs, of its K and transposed V tiles: padded so a fragment read hits 32 banks. */
    static final int MMA_K_PITCH = MMA_HEAD_DIM / 2 + 4;
    static final int MMA_V_PITCH = MMA_KEYS / 2 + 4;
    /** A masked score of {@link #timeAttentionMma}, and the bound above which a score is a real one. */
    static final float MASKED = -3.4e38f;
    static final float VALID = -3.0e38f;

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

    /** {@link #qkNormRope} on {@code q|k|v} projections held in FP16, in place. */
    public static void qkNormRopeHalf(KernelContext ctx, HalfFloatArray qkv, FloatArray w, int bias, int qNorm, int kNorm, int invFreq,
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
                a = qkv.get(base + 2 * i).getFloat32();
                b = qkv.get(base + 2 * i + 1).getFloat32();
                if (bias >= 0) {
                    int bo = bias + which * d + head * hd + 2 * i;
                    a += w.get(bo);
                    b += w.get(bo + 1);
                    int v = tok * 3 * d + 2 * d + head * hd + tid;
                    setHalf(qkv, v, qkv.get(v).getFloat32() + w.get(bias + 2 * d + head * hd + tid));
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
                setHalf(qkv, base + 2 * i, na);
                setHalf(qkv, base + 2 * i + 1, nb);
            }
        }
    }

    /**
     * Attention along each row's patches per (row, head): a valid query attends to the valid keys up to itself,
     * or to all of them in a known-future covariate's row; an all-padding query gets zeros. Flash-style: one
     * work-group of {@link #ATTN_THREADS} per (row, head, {@link #ATTN_QUERIES} queries) walks the keys in
     * blocks of {@link #ATTN_KEYS} with an online softmax. Each block's keys are staged transposed with a
     * padded stride, so a warp's score dot products read distinct banks, then its values in the same buffer;
     * the queries (padded likewise), scores and output accumulators also sit in local memory. Also writes the
     * FP16 copy {@code th} of {@code out} when {@code twin} is 1.
     */
    public static void timeAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray valid, IntArray future, int rows,
            int n, int heads, int hd, int d, float scale, HalfFloatArray th, int twin) {
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
                    store(out, th, twin, (row0 + t) * d + head * hd + dd, dn > 0f ? acc[qi * HEAD_DIM + dd] / dn : 0f);
                }
            }
        }
    }

    /**
     * {@link #timeAttention} on tensor cores with warp-level {@code mma} (CUDA only, compute capability
     * 8.0+), flash attention in the FlashAttention-2 layout: one work-group of {@link #MMA_THREADS}
     * threads per (row, head, {@link #MMA_QUERIES} consecutive patches), each warp 16 of the queries.
     * Each lane packs its Q fragments from {@code qkv} once; per block of {@link #MMA_KEYS} keys the
     * work-group stages K and V in shared memory as FP16 pairs (V transposed), each warp computes its
     * 16 × {@link #MMA_KEYS} scores with {@code mma}, masks them as {@link #timeAttention} does, folds
     * them into its online softmax in registers and feeds the probabilities straight back as the A
     * fragments of P·V. The output accumulates in FP32 registers. Heads must be {@link #MMA_HEAD_DIM}
     * wide. Writes {@code out} as {@link #timeAttention} does, and its FP16 copy {@code th} when
     * {@code twin} is 1.
     */
    public static void timeAttentionMma(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray valid, IntArray future,
            int rows, int n, int heads, int d, float scale, HalfFloatArray th, int twin) {
        int qTiles = (n + MMA_QUERIES - 1) / MMA_QUERIES;
        if (ctx.groupIdx < rows * heads * qTiles) {
            int[] ks = ctx.allocateIntLocalArray(MMA_KEYS * MMA_K_PITCH);
            int[] vs = ctx.allocateIntLocalArray(MMA_HEAD_DIM * MMA_V_PITCH);
            int g = ctx.groupIdx;
            int rh = g / qTiles;
            int qt = g - rh * qTiles;
            int r = rh / heads;
            int hh = rh - r * heads;
            int tid = ctx.localIdx;
            int lane = tid & 31;
            int quad = lane & 3;
            int ra = qt * MMA_QUERIES + (tid >> 5) * 16 + (lane >> 2);
            int rb = ra + 8;
            int c = quad * 2;
            int row0 = r * n;
            int qBase = row0 * 3 * d + hh * MMA_HEAD_DIM;
            int qa = qBase + ra * 3 * d;
            int qb = qBase + rb * 3 * d;
            boolean okA = ra < n;
            boolean okB = rb < n;
            int vA = 0;
            int vB = 0;
            if (okA) {
                vA = valid.get(row0 + ra);
            }
            if (okB) {
                vB = valid.get(row0 + rb);
            }
            int bidir = future.get(r);
            int lastQ = TornadoMath.min(qt * MMA_QUERIES + MMA_QUERIES, n) - 1;
            int kEnd = n;
            if (bidir == 0) {
                kEnd = lastQ + 1;
            }
            float qx00 = okA ? qkv.get(qa + 0 + c) : 0f;
            float qy00 = okA ? qkv.get(qa + 0 + c + 1) : 0f;
            int q00 = pack(qx00, qy00);
            float qx01 = okB ? qkv.get(qb + 0 + c) : 0f;
            float qy01 = okB ? qkv.get(qb + 0 + c + 1) : 0f;
            int q01 = pack(qx01, qy01);
            float qx02 = okA ? qkv.get(qa + 8 + c) : 0f;
            float qy02 = okA ? qkv.get(qa + 8 + c + 1) : 0f;
            int q02 = pack(qx02, qy02);
            float qx03 = okB ? qkv.get(qb + 8 + c) : 0f;
            float qy03 = okB ? qkv.get(qb + 8 + c + 1) : 0f;
            int q03 = pack(qx03, qy03);
            float qx10 = okA ? qkv.get(qa + 16 + c) : 0f;
            float qy10 = okA ? qkv.get(qa + 16 + c + 1) : 0f;
            int q10 = pack(qx10, qy10);
            float qx11 = okB ? qkv.get(qb + 16 + c) : 0f;
            float qy11 = okB ? qkv.get(qb + 16 + c + 1) : 0f;
            int q11 = pack(qx11, qy11);
            float qx12 = okA ? qkv.get(qa + 24 + c) : 0f;
            float qy12 = okA ? qkv.get(qa + 24 + c + 1) : 0f;
            int q12 = pack(qx12, qy12);
            float qx13 = okB ? qkv.get(qb + 24 + c) : 0f;
            float qy13 = okB ? qkv.get(qb + 24 + c + 1) : 0f;
            int q13 = pack(qx13, qy13);
            float qx20 = okA ? qkv.get(qa + 32 + c) : 0f;
            float qy20 = okA ? qkv.get(qa + 32 + c + 1) : 0f;
            int q20 = pack(qx20, qy20);
            float qx21 = okB ? qkv.get(qb + 32 + c) : 0f;
            float qy21 = okB ? qkv.get(qb + 32 + c + 1) : 0f;
            int q21 = pack(qx21, qy21);
            float qx22 = okA ? qkv.get(qa + 40 + c) : 0f;
            float qy22 = okA ? qkv.get(qa + 40 + c + 1) : 0f;
            int q22 = pack(qx22, qy22);
            float qx23 = okB ? qkv.get(qb + 40 + c) : 0f;
            float qy23 = okB ? qkv.get(qb + 40 + c + 1) : 0f;
            int q23 = pack(qx23, qy23);
            float qx30 = okA ? qkv.get(qa + 48 + c) : 0f;
            float qy30 = okA ? qkv.get(qa + 48 + c + 1) : 0f;
            int q30 = pack(qx30, qy30);
            float qx31 = okB ? qkv.get(qb + 48 + c) : 0f;
            float qy31 = okB ? qkv.get(qb + 48 + c + 1) : 0f;
            int q31 = pack(qx31, qy31);
            float qx32 = okA ? qkv.get(qa + 56 + c) : 0f;
            float qy32 = okA ? qkv.get(qa + 56 + c + 1) : 0f;
            int q32 = pack(qx32, qy32);
            float qx33 = okB ? qkv.get(qb + 56 + c) : 0f;
            float qy33 = okB ? qkv.get(qb + 56 + c + 1) : 0f;
            int q33 = pack(qx33, qy33);
            float qx40 = okA ? qkv.get(qa + 64 + c) : 0f;
            float qy40 = okA ? qkv.get(qa + 64 + c + 1) : 0f;
            int q40 = pack(qx40, qy40);
            float qx41 = okB ? qkv.get(qb + 64 + c) : 0f;
            float qy41 = okB ? qkv.get(qb + 64 + c + 1) : 0f;
            int q41 = pack(qx41, qy41);
            float qx42 = okA ? qkv.get(qa + 72 + c) : 0f;
            float qy42 = okA ? qkv.get(qa + 72 + c + 1) : 0f;
            int q42 = pack(qx42, qy42);
            float qx43 = okB ? qkv.get(qb + 72 + c) : 0f;
            float qy43 = okB ? qkv.get(qb + 72 + c + 1) : 0f;
            int q43 = pack(qx43, qy43);
            float qx50 = okA ? qkv.get(qa + 80 + c) : 0f;
            float qy50 = okA ? qkv.get(qa + 80 + c + 1) : 0f;
            int q50 = pack(qx50, qy50);
            float qx51 = okB ? qkv.get(qb + 80 + c) : 0f;
            float qy51 = okB ? qkv.get(qb + 80 + c + 1) : 0f;
            int q51 = pack(qx51, qy51);
            float qx52 = okA ? qkv.get(qa + 88 + c) : 0f;
            float qy52 = okA ? qkv.get(qa + 88 + c + 1) : 0f;
            int q52 = pack(qx52, qy52);
            float qx53 = okB ? qkv.get(qb + 88 + c) : 0f;
            float qy53 = okB ? qkv.get(qb + 88 + c + 1) : 0f;
            int q53 = pack(qx53, qy53);
            float qx60 = okA ? qkv.get(qa + 96 + c) : 0f;
            float qy60 = okA ? qkv.get(qa + 96 + c + 1) : 0f;
            int q60 = pack(qx60, qy60);
            float qx61 = okB ? qkv.get(qb + 96 + c) : 0f;
            float qy61 = okB ? qkv.get(qb + 96 + c + 1) : 0f;
            int q61 = pack(qx61, qy61);
            float qx62 = okA ? qkv.get(qa + 104 + c) : 0f;
            float qy62 = okA ? qkv.get(qa + 104 + c + 1) : 0f;
            int q62 = pack(qx62, qy62);
            float qx63 = okB ? qkv.get(qb + 104 + c) : 0f;
            float qy63 = okB ? qkv.get(qb + 104 + c + 1) : 0f;
            int q63 = pack(qx63, qy63);
            float qx70 = okA ? qkv.get(qa + 112 + c) : 0f;
            float qy70 = okA ? qkv.get(qa + 112 + c + 1) : 0f;
            int q70 = pack(qx70, qy70);
            float qx71 = okB ? qkv.get(qb + 112 + c) : 0f;
            float qy71 = okB ? qkv.get(qb + 112 + c + 1) : 0f;
            int q71 = pack(qx71, qy71);
            float qx72 = okA ? qkv.get(qa + 120 + c) : 0f;
            float qy72 = okA ? qkv.get(qa + 120 + c + 1) : 0f;
            int q72 = pack(qx72, qy72);
            float qx73 = okB ? qkv.get(qb + 120 + c) : 0f;
            float qy73 = okB ? qkv.get(qb + 120 + c + 1) : 0f;
            int q73 = pack(qx73, qy73);
            float mA = MASKED;
            float mB = MASKED;
            float lA = 0f;
            float lB = 0f;
            float o0_0 = 0f, o0_1 = 0f, o0_2 = 0f, o0_3 = 0f;
            float o1_0 = 0f, o1_1 = 0f, o1_2 = 0f, o1_3 = 0f;
            float o2_0 = 0f, o2_1 = 0f, o2_2 = 0f, o2_3 = 0f;
            float o3_0 = 0f, o3_1 = 0f, o3_2 = 0f, o3_3 = 0f;
            float o4_0 = 0f, o4_1 = 0f, o4_2 = 0f, o4_3 = 0f;
            float o5_0 = 0f, o5_1 = 0f, o5_2 = 0f, o5_3 = 0f;
            float o6_0 = 0f, o6_1 = 0f, o6_2 = 0f, o6_3 = 0f;
            float o7_0 = 0f, o7_1 = 0f, o7_2 = 0f, o7_3 = 0f;
            float o8_0 = 0f, o8_1 = 0f, o8_2 = 0f, o8_3 = 0f;
            float o9_0 = 0f, o9_1 = 0f, o9_2 = 0f, o9_3 = 0f;
            float o10_0 = 0f, o10_1 = 0f, o10_2 = 0f, o10_3 = 0f;
            float o11_0 = 0f, o11_1 = 0f, o11_2 = 0f, o11_3 = 0f;
            float o12_0 = 0f, o12_1 = 0f, o12_2 = 0f, o12_3 = 0f;
            float o13_0 = 0f, o13_1 = 0f, o13_2 = 0f, o13_3 = 0f;
            float o14_0 = 0f, o14_1 = 0f, o14_2 = 0f, o14_3 = 0f;
            float o15_0 = 0f, o15_1 = 0f, o15_2 = 0f, o15_3 = 0f;
            for (int k0 = 0; k0 < kEnd; k0 += MMA_KEYS) {
                // K: key j, dimensions 2p, 2p + 1 at ks[j·pitch + p]; V transposed: dimension e, keys 2p, 2p + 1 at vs[e·pitch + p]
                for (int i = tid; i < MMA_KEYS * MMA_HEAD_DIM / 2; i += MMA_THREADS) {
                    int j = i >> 6;
                    int p = i & 63;
                    int ko = qBase + (k0 + j) * 3 * d + d + 2 * p;
                    boolean in = k0 + j < n;
                    float k0v = in ? qkv.get(ko) : 0f;
                    float k1v = in ? qkv.get(ko + 1) : 0f;
                    ks[j * MMA_K_PITCH + p] = pack(k0v, k1v);
                }
                for (int i = tid; i < MMA_KEYS * MMA_HEAD_DIM / 2; i += MMA_THREADS) {
                    int e = i & (MMA_HEAD_DIM - 1);
                    int p = i >> 7;
                    int vo = qBase + (k0 + 2 * p) * 3 * d + 2 * d + e;
                    float v0 = k0 + 2 * p < n ? qkv.get(vo) : 0f;
                    float v1 = k0 + 2 * p + 1 < n ? qkv.get(vo + 3 * d) : 0f;
                    vs[e * MMA_V_PITCH + p] = pack(v0, v1);
                }
                ctx.localBarrier();
                HalfFloat[] fq0 = ctx.mmaFragmentA(q00, q01, q02, q03);
                HalfFloat[] fq1 = ctx.mmaFragmentA(q10, q11, q12, q13);
                HalfFloat[] fq2 = ctx.mmaFragmentA(q20, q21, q22, q23);
                HalfFloat[] fq3 = ctx.mmaFragmentA(q30, q31, q32, q33);
                HalfFloat[] fq4 = ctx.mmaFragmentA(q40, q41, q42, q43);
                HalfFloat[] fq5 = ctx.mmaFragmentA(q50, q51, q52, q53);
                HalfFloat[] fq6 = ctx.mmaFragmentA(q60, q61, q62, q63);
                HalfFloat[] fq7 = ctx.mmaFragmentA(q70, q71, q72, q73);
                float[] s0 = ctx.mmaFragment(0f);
                int kr0 = (0 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk0_0 = ctx.mmaFragmentB(ks[kr0 + 0], ks[kr0 + 4]);
                s0 = ctx.mma(fq0, fk0_0, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_1 = ctx.mmaFragmentB(ks[kr0 + 8], ks[kr0 + 12]);
                s0 = ctx.mma(fq1, fk0_1, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_2 = ctx.mmaFragmentB(ks[kr0 + 16], ks[kr0 + 20]);
                s0 = ctx.mma(fq2, fk0_2, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_3 = ctx.mmaFragmentB(ks[kr0 + 24], ks[kr0 + 28]);
                s0 = ctx.mma(fq3, fk0_3, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_4 = ctx.mmaFragmentB(ks[kr0 + 32], ks[kr0 + 36]);
                s0 = ctx.mma(fq4, fk0_4, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_5 = ctx.mmaFragmentB(ks[kr0 + 40], ks[kr0 + 44]);
                s0 = ctx.mma(fq5, fk0_5, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_6 = ctx.mmaFragmentB(ks[kr0 + 48], ks[kr0 + 52]);
                s0 = ctx.mma(fq6, fk0_6, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_7 = ctx.mmaFragmentB(ks[kr0 + 56], ks[kr0 + 60]);
                s0 = ctx.mma(fq7, fk0_7, s0, MMAShape.M16N8K16);
                float[] s1 = ctx.mmaFragment(0f);
                int kr1 = (8 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk1_0 = ctx.mmaFragmentB(ks[kr1 + 0], ks[kr1 + 4]);
                s1 = ctx.mma(fq0, fk1_0, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_1 = ctx.mmaFragmentB(ks[kr1 + 8], ks[kr1 + 12]);
                s1 = ctx.mma(fq1, fk1_1, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_2 = ctx.mmaFragmentB(ks[kr1 + 16], ks[kr1 + 20]);
                s1 = ctx.mma(fq2, fk1_2, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_3 = ctx.mmaFragmentB(ks[kr1 + 24], ks[kr1 + 28]);
                s1 = ctx.mma(fq3, fk1_3, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_4 = ctx.mmaFragmentB(ks[kr1 + 32], ks[kr1 + 36]);
                s1 = ctx.mma(fq4, fk1_4, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_5 = ctx.mmaFragmentB(ks[kr1 + 40], ks[kr1 + 44]);
                s1 = ctx.mma(fq5, fk1_5, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_6 = ctx.mmaFragmentB(ks[kr1 + 48], ks[kr1 + 52]);
                s1 = ctx.mma(fq6, fk1_6, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_7 = ctx.mmaFragmentB(ks[kr1 + 56], ks[kr1 + 60]);
                s1 = ctx.mma(fq7, fk1_7, s1, MMAShape.M16N8K16);
                float[] s2 = ctx.mmaFragment(0f);
                int kr2 = (16 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk2_0 = ctx.mmaFragmentB(ks[kr2 + 0], ks[kr2 + 4]);
                s2 = ctx.mma(fq0, fk2_0, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_1 = ctx.mmaFragmentB(ks[kr2 + 8], ks[kr2 + 12]);
                s2 = ctx.mma(fq1, fk2_1, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_2 = ctx.mmaFragmentB(ks[kr2 + 16], ks[kr2 + 20]);
                s2 = ctx.mma(fq2, fk2_2, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_3 = ctx.mmaFragmentB(ks[kr2 + 24], ks[kr2 + 28]);
                s2 = ctx.mma(fq3, fk2_3, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_4 = ctx.mmaFragmentB(ks[kr2 + 32], ks[kr2 + 36]);
                s2 = ctx.mma(fq4, fk2_4, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_5 = ctx.mmaFragmentB(ks[kr2 + 40], ks[kr2 + 44]);
                s2 = ctx.mma(fq5, fk2_5, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_6 = ctx.mmaFragmentB(ks[kr2 + 48], ks[kr2 + 52]);
                s2 = ctx.mma(fq6, fk2_6, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_7 = ctx.mmaFragmentB(ks[kr2 + 56], ks[kr2 + 60]);
                s2 = ctx.mma(fq7, fk2_7, s2, MMAShape.M16N8K16);
                float[] s3 = ctx.mmaFragment(0f);
                int kr3 = (24 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk3_0 = ctx.mmaFragmentB(ks[kr3 + 0], ks[kr3 + 4]);
                s3 = ctx.mma(fq0, fk3_0, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_1 = ctx.mmaFragmentB(ks[kr3 + 8], ks[kr3 + 12]);
                s3 = ctx.mma(fq1, fk3_1, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_2 = ctx.mmaFragmentB(ks[kr3 + 16], ks[kr3 + 20]);
                s3 = ctx.mma(fq2, fk3_2, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_3 = ctx.mmaFragmentB(ks[kr3 + 24], ks[kr3 + 28]);
                s3 = ctx.mma(fq3, fk3_3, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_4 = ctx.mmaFragmentB(ks[kr3 + 32], ks[kr3 + 36]);
                s3 = ctx.mma(fq4, fk3_4, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_5 = ctx.mmaFragmentB(ks[kr3 + 40], ks[kr3 + 44]);
                s3 = ctx.mma(fq5, fk3_5, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_6 = ctx.mmaFragmentB(ks[kr3 + 48], ks[kr3 + 52]);
                s3 = ctx.mma(fq6, fk3_6, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_7 = ctx.mmaFragmentB(ks[kr3 + 56], ks[kr3 + 60]);
                s3 = ctx.mma(fq7, fk3_7, s3, MMAShape.M16N8K16);
                int kv00 = keyValid(valid, row0, n, k0 + 0 + c);
                float x00 = allowed(ra, k0 + 0 + c, vA, kv00, bidir) != 0 ? s0[0] * scale : MASKED;
                float y00 = allowed(rb, k0 + 0 + c, vB, kv00, bidir) != 0 ? s0[2] * scale : MASKED;
                int kv01 = keyValid(valid, row0, n, k0 + 1 + c);
                float x01 = allowed(ra, k0 + 1 + c, vA, kv01, bidir) != 0 ? s0[1] * scale : MASKED;
                float y01 = allowed(rb, k0 + 1 + c, vB, kv01, bidir) != 0 ? s0[3] * scale : MASKED;
                int kv10 = keyValid(valid, row0, n, k0 + 8 + c);
                float x10 = allowed(ra, k0 + 8 + c, vA, kv10, bidir) != 0 ? s1[0] * scale : MASKED;
                float y10 = allowed(rb, k0 + 8 + c, vB, kv10, bidir) != 0 ? s1[2] * scale : MASKED;
                int kv11 = keyValid(valid, row0, n, k0 + 9 + c);
                float x11 = allowed(ra, k0 + 9 + c, vA, kv11, bidir) != 0 ? s1[1] * scale : MASKED;
                float y11 = allowed(rb, k0 + 9 + c, vB, kv11, bidir) != 0 ? s1[3] * scale : MASKED;
                int kv20 = keyValid(valid, row0, n, k0 + 16 + c);
                float x20 = allowed(ra, k0 + 16 + c, vA, kv20, bidir) != 0 ? s2[0] * scale : MASKED;
                float y20 = allowed(rb, k0 + 16 + c, vB, kv20, bidir) != 0 ? s2[2] * scale : MASKED;
                int kv21 = keyValid(valid, row0, n, k0 + 17 + c);
                float x21 = allowed(ra, k0 + 17 + c, vA, kv21, bidir) != 0 ? s2[1] * scale : MASKED;
                float y21 = allowed(rb, k0 + 17 + c, vB, kv21, bidir) != 0 ? s2[3] * scale : MASKED;
                int kv30 = keyValid(valid, row0, n, k0 + 24 + c);
                float x30 = allowed(ra, k0 + 24 + c, vA, kv30, bidir) != 0 ? s3[0] * scale : MASKED;
                float y30 = allowed(rb, k0 + 24 + c, vB, kv30, bidir) != 0 ? s3[2] * scale : MASKED;
                int kv31 = keyValid(valid, row0, n, k0 + 25 + c);
                float x31 = allowed(ra, k0 + 25 + c, vA, kv31, bidir) != 0 ? s3[1] * scale : MASKED;
                float y31 = allowed(rb, k0 + 25 + c, vB, kv31, bidir) != 0 ? s3[3] * scale : MASKED;
                float bA = TornadoMath.max(TornadoMath.max(TornadoMath.max(x00, x01), TornadoMath.max(x10, x11)), TornadoMath.max(TornadoMath.max(x20, x21), TornadoMath.max(x30, x31)));
                float bB = TornadoMath.max(TornadoMath.max(TornadoMath.max(y00, y01), TornadoMath.max(y10, y11)), TornadoMath.max(TornadoMath.max(y20, y21), TornadoMath.max(y30, y31)));
                bA = TornadoMath.max(bA, ctx.simdShuffle(bA, lane ^ 1));
                bA = TornadoMath.max(bA, ctx.simdShuffle(bA, lane ^ 2));
                bB = TornadoMath.max(bB, ctx.simdShuffle(bB, lane ^ 1));
                bB = TornadoMath.max(bB, ctx.simdShuffle(bB, lane ^ 2));
                float nA = TornadoMath.max(mA, bA);
                float nB = TornadoMath.max(mB, bB);
                float cA = bA > VALID ? TornadoMath.exp(mA - nA) : 1f;
                float cB = bB > VALID ? TornadoMath.exp(mB - nB) : 1f;
                float pa00 = x00 > VALID ? TornadoMath.exp(x00 - nA) : 0f;
                float pb00 = y00 > VALID ? TornadoMath.exp(y00 - nB) : 0f;
                float pa01 = x01 > VALID ? TornadoMath.exp(x01 - nA) : 0f;
                float pb01 = y01 > VALID ? TornadoMath.exp(y01 - nB) : 0f;
                float pa10 = x10 > VALID ? TornadoMath.exp(x10 - nA) : 0f;
                float pb10 = y10 > VALID ? TornadoMath.exp(y10 - nB) : 0f;
                float pa11 = x11 > VALID ? TornadoMath.exp(x11 - nA) : 0f;
                float pb11 = y11 > VALID ? TornadoMath.exp(y11 - nB) : 0f;
                float pa20 = x20 > VALID ? TornadoMath.exp(x20 - nA) : 0f;
                float pb20 = y20 > VALID ? TornadoMath.exp(y20 - nB) : 0f;
                float pa21 = x21 > VALID ? TornadoMath.exp(x21 - nA) : 0f;
                float pb21 = y21 > VALID ? TornadoMath.exp(y21 - nB) : 0f;
                float pa30 = x30 > VALID ? TornadoMath.exp(x30 - nA) : 0f;
                float pb30 = y30 > VALID ? TornadoMath.exp(y30 - nB) : 0f;
                float pa31 = x31 > VALID ? TornadoMath.exp(x31 - nA) : 0f;
                float pb31 = y31 > VALID ? TornadoMath.exp(y31 - nB) : 0f;
                lA = lA * cA + pa00 + pa01 + pa10 + pa11 + pa20 + pa21 + pa30 + pa31;
                lB = lB * cB + pb00 + pb01 + pb10 + pb11 + pb20 + pb21 + pb30 + pb31;
                mA = nA;
                mB = nB;
                HalfFloat[] fp0 = ctx.mmaFragmentA(pack(pa00, pa01), pack(pb00, pb01), pack(pa10, pa11), pack(pb10, pb11));
                HalfFloat[] fp1 = ctx.mmaFragmentA(pack(pa20, pa21), pack(pb20, pb21), pack(pa30, pa31), pack(pb30, pb31));
                int vr0 = (0 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t0 = ctx.mmaFragment(0f);
                t0 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr0 + 0], vs[vr0 + 4]), t0, MMAShape.M16N8K16);
                t0 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr0 + 8], vs[vr0 + 12]), t0, MMAShape.M16N8K16);
                o0_0 = o0_0 * cA + t0[0];
                o0_1 = o0_1 * cA + t0[1];
                o0_2 = o0_2 * cB + t0[2];
                o0_3 = o0_3 * cB + t0[3];
                int vr1 = (8 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t1 = ctx.mmaFragment(0f);
                t1 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr1 + 0], vs[vr1 + 4]), t1, MMAShape.M16N8K16);
                t1 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr1 + 8], vs[vr1 + 12]), t1, MMAShape.M16N8K16);
                o1_0 = o1_0 * cA + t1[0];
                o1_1 = o1_1 * cA + t1[1];
                o1_2 = o1_2 * cB + t1[2];
                o1_3 = o1_3 * cB + t1[3];
                int vr2 = (16 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t2 = ctx.mmaFragment(0f);
                t2 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr2 + 0], vs[vr2 + 4]), t2, MMAShape.M16N8K16);
                t2 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr2 + 8], vs[vr2 + 12]), t2, MMAShape.M16N8K16);
                o2_0 = o2_0 * cA + t2[0];
                o2_1 = o2_1 * cA + t2[1];
                o2_2 = o2_2 * cB + t2[2];
                o2_3 = o2_3 * cB + t2[3];
                int vr3 = (24 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t3 = ctx.mmaFragment(0f);
                t3 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr3 + 0], vs[vr3 + 4]), t3, MMAShape.M16N8K16);
                t3 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr3 + 8], vs[vr3 + 12]), t3, MMAShape.M16N8K16);
                o3_0 = o3_0 * cA + t3[0];
                o3_1 = o3_1 * cA + t3[1];
                o3_2 = o3_2 * cB + t3[2];
                o3_3 = o3_3 * cB + t3[3];
                int vr4 = (32 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t4 = ctx.mmaFragment(0f);
                t4 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr4 + 0], vs[vr4 + 4]), t4, MMAShape.M16N8K16);
                t4 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr4 + 8], vs[vr4 + 12]), t4, MMAShape.M16N8K16);
                o4_0 = o4_0 * cA + t4[0];
                o4_1 = o4_1 * cA + t4[1];
                o4_2 = o4_2 * cB + t4[2];
                o4_3 = o4_3 * cB + t4[3];
                int vr5 = (40 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t5 = ctx.mmaFragment(0f);
                t5 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr5 + 0], vs[vr5 + 4]), t5, MMAShape.M16N8K16);
                t5 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr5 + 8], vs[vr5 + 12]), t5, MMAShape.M16N8K16);
                o5_0 = o5_0 * cA + t5[0];
                o5_1 = o5_1 * cA + t5[1];
                o5_2 = o5_2 * cB + t5[2];
                o5_3 = o5_3 * cB + t5[3];
                int vr6 = (48 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t6 = ctx.mmaFragment(0f);
                t6 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr6 + 0], vs[vr6 + 4]), t6, MMAShape.M16N8K16);
                t6 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr6 + 8], vs[vr6 + 12]), t6, MMAShape.M16N8K16);
                o6_0 = o6_0 * cA + t6[0];
                o6_1 = o6_1 * cA + t6[1];
                o6_2 = o6_2 * cB + t6[2];
                o6_3 = o6_3 * cB + t6[3];
                int vr7 = (56 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t7 = ctx.mmaFragment(0f);
                t7 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr7 + 0], vs[vr7 + 4]), t7, MMAShape.M16N8K16);
                t7 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr7 + 8], vs[vr7 + 12]), t7, MMAShape.M16N8K16);
                o7_0 = o7_0 * cA + t7[0];
                o7_1 = o7_1 * cA + t7[1];
                o7_2 = o7_2 * cB + t7[2];
                o7_3 = o7_3 * cB + t7[3];
                int vr8 = (64 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t8 = ctx.mmaFragment(0f);
                t8 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr8 + 0], vs[vr8 + 4]), t8, MMAShape.M16N8K16);
                t8 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr8 + 8], vs[vr8 + 12]), t8, MMAShape.M16N8K16);
                o8_0 = o8_0 * cA + t8[0];
                o8_1 = o8_1 * cA + t8[1];
                o8_2 = o8_2 * cB + t8[2];
                o8_3 = o8_3 * cB + t8[3];
                int vr9 = (72 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t9 = ctx.mmaFragment(0f);
                t9 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr9 + 0], vs[vr9 + 4]), t9, MMAShape.M16N8K16);
                t9 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr9 + 8], vs[vr9 + 12]), t9, MMAShape.M16N8K16);
                o9_0 = o9_0 * cA + t9[0];
                o9_1 = o9_1 * cA + t9[1];
                o9_2 = o9_2 * cB + t9[2];
                o9_3 = o9_3 * cB + t9[3];
                int vr10 = (80 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t10 = ctx.mmaFragment(0f);
                t10 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr10 + 0], vs[vr10 + 4]), t10, MMAShape.M16N8K16);
                t10 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr10 + 8], vs[vr10 + 12]), t10, MMAShape.M16N8K16);
                o10_0 = o10_0 * cA + t10[0];
                o10_1 = o10_1 * cA + t10[1];
                o10_2 = o10_2 * cB + t10[2];
                o10_3 = o10_3 * cB + t10[3];
                int vr11 = (88 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t11 = ctx.mmaFragment(0f);
                t11 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr11 + 0], vs[vr11 + 4]), t11, MMAShape.M16N8K16);
                t11 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr11 + 8], vs[vr11 + 12]), t11, MMAShape.M16N8K16);
                o11_0 = o11_0 * cA + t11[0];
                o11_1 = o11_1 * cA + t11[1];
                o11_2 = o11_2 * cB + t11[2];
                o11_3 = o11_3 * cB + t11[3];
                int vr12 = (96 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t12 = ctx.mmaFragment(0f);
                t12 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr12 + 0], vs[vr12 + 4]), t12, MMAShape.M16N8K16);
                t12 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr12 + 8], vs[vr12 + 12]), t12, MMAShape.M16N8K16);
                o12_0 = o12_0 * cA + t12[0];
                o12_1 = o12_1 * cA + t12[1];
                o12_2 = o12_2 * cB + t12[2];
                o12_3 = o12_3 * cB + t12[3];
                int vr13 = (104 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t13 = ctx.mmaFragment(0f);
                t13 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr13 + 0], vs[vr13 + 4]), t13, MMAShape.M16N8K16);
                t13 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr13 + 8], vs[vr13 + 12]), t13, MMAShape.M16N8K16);
                o13_0 = o13_0 * cA + t13[0];
                o13_1 = o13_1 * cA + t13[1];
                o13_2 = o13_2 * cB + t13[2];
                o13_3 = o13_3 * cB + t13[3];
                int vr14 = (112 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t14 = ctx.mmaFragment(0f);
                t14 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr14 + 0], vs[vr14 + 4]), t14, MMAShape.M16N8K16);
                t14 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr14 + 8], vs[vr14 + 12]), t14, MMAShape.M16N8K16);
                o14_0 = o14_0 * cA + t14[0];
                o14_1 = o14_1 * cA + t14[1];
                o14_2 = o14_2 * cB + t14[2];
                o14_3 = o14_3 * cB + t14[3];
                int vr15 = (120 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t15 = ctx.mmaFragment(0f);
                t15 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr15 + 0], vs[vr15 + 4]), t15, MMAShape.M16N8K16);
                t15 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr15 + 8], vs[vr15 + 12]), t15, MMAShape.M16N8K16);
                o15_0 = o15_0 * cA + t15[0];
                o15_1 = o15_1 * cA + t15[1];
                o15_2 = o15_2 * cB + t15[2];
                o15_3 = o15_3 * cB + t15[3];
                ctx.localBarrier();
            }
            lA = lA + ctx.simdShuffle(lA, lane ^ 1);
            lA = lA + ctx.simdShuffle(lA, lane ^ 2);
            lB = lB + ctx.simdShuffle(lB, lane ^ 1);
            lB = lB + ctx.simdShuffle(lB, lane ^ 2);
            // an all-padding query reads no key and gets zeros
            float iA = lA > 0f ? 1f / lA : 0f;
            float iB = lB > 0f ? 1f / lB : 0f;
            int oa = (row0 + ra) * d + hh * MMA_HEAD_DIM + c;
            int ob = (row0 + rb) * d + hh * MMA_HEAD_DIM + c;
            if (okA) {
                store(out, th, twin, oa + 0, o0_0 * iA);
                store(out, th, twin, oa + 1, o0_1 * iA);
                store(out, th, twin, oa + 8, o1_0 * iA);
                store(out, th, twin, oa + 9, o1_1 * iA);
                store(out, th, twin, oa + 16, o2_0 * iA);
                store(out, th, twin, oa + 17, o2_1 * iA);
                store(out, th, twin, oa + 24, o3_0 * iA);
                store(out, th, twin, oa + 25, o3_1 * iA);
                store(out, th, twin, oa + 32, o4_0 * iA);
                store(out, th, twin, oa + 33, o4_1 * iA);
                store(out, th, twin, oa + 40, o5_0 * iA);
                store(out, th, twin, oa + 41, o5_1 * iA);
                store(out, th, twin, oa + 48, o6_0 * iA);
                store(out, th, twin, oa + 49, o6_1 * iA);
                store(out, th, twin, oa + 56, o7_0 * iA);
                store(out, th, twin, oa + 57, o7_1 * iA);
                store(out, th, twin, oa + 64, o8_0 * iA);
                store(out, th, twin, oa + 65, o8_1 * iA);
                store(out, th, twin, oa + 72, o9_0 * iA);
                store(out, th, twin, oa + 73, o9_1 * iA);
                store(out, th, twin, oa + 80, o10_0 * iA);
                store(out, th, twin, oa + 81, o10_1 * iA);
                store(out, th, twin, oa + 88, o11_0 * iA);
                store(out, th, twin, oa + 89, o11_1 * iA);
                store(out, th, twin, oa + 96, o12_0 * iA);
                store(out, th, twin, oa + 97, o12_1 * iA);
                store(out, th, twin, oa + 104, o13_0 * iA);
                store(out, th, twin, oa + 105, o13_1 * iA);
                store(out, th, twin, oa + 112, o14_0 * iA);
                store(out, th, twin, oa + 113, o14_1 * iA);
                store(out, th, twin, oa + 120, o15_0 * iA);
                store(out, th, twin, oa + 121, o15_1 * iA);
            }
            if (okB) {
                store(out, th, twin, ob + 0, o0_2 * iB);
                store(out, th, twin, ob + 1, o0_3 * iB);
                store(out, th, twin, ob + 8, o1_2 * iB);
                store(out, th, twin, ob + 9, o1_3 * iB);
                store(out, th, twin, ob + 16, o2_2 * iB);
                store(out, th, twin, ob + 17, o2_3 * iB);
                store(out, th, twin, ob + 24, o3_2 * iB);
                store(out, th, twin, ob + 25, o3_3 * iB);
                store(out, th, twin, ob + 32, o4_2 * iB);
                store(out, th, twin, ob + 33, o4_3 * iB);
                store(out, th, twin, ob + 40, o5_2 * iB);
                store(out, th, twin, ob + 41, o5_3 * iB);
                store(out, th, twin, ob + 48, o6_2 * iB);
                store(out, th, twin, ob + 49, o6_3 * iB);
                store(out, th, twin, ob + 56, o7_2 * iB);
                store(out, th, twin, ob + 57, o7_3 * iB);
                store(out, th, twin, ob + 64, o8_2 * iB);
                store(out, th, twin, ob + 65, o8_3 * iB);
                store(out, th, twin, ob + 72, o9_2 * iB);
                store(out, th, twin, ob + 73, o9_3 * iB);
                store(out, th, twin, ob + 80, o10_2 * iB);
                store(out, th, twin, ob + 81, o10_3 * iB);
                store(out, th, twin, ob + 88, o11_2 * iB);
                store(out, th, twin, ob + 89, o11_3 * iB);
                store(out, th, twin, ob + 96, o12_2 * iB);
                store(out, th, twin, ob + 97, o12_3 * iB);
                store(out, th, twin, ob + 104, o13_2 * iB);
                store(out, th, twin, ob + 105, o13_3 * iB);
                store(out, th, twin, ob + 112, o14_2 * iB);
                store(out, th, twin, ob + 113, o14_3 * iB);
                store(out, th, twin, ob + 120, o15_2 * iB);
                store(out, th, twin, ob + 121, o15_3 * iB);
            }
        }
    }

    /**
     * {@link #timeAttentionMma} on {@code q|k|v} projections held in FP16 ({@link #qkNormRopeHalf}): the
     * fragments take the stored half pairs as they are.
     */
    public static void timeAttentionMmaHalf(KernelContext ctx, HalfFloatArray qkv, FloatArray out, IntArray valid, IntArray future,
            int rows, int n, int heads, int d, float scale, HalfFloatArray th, int twin) {
        int qTiles = (n + MMA_QUERIES - 1) / MMA_QUERIES;
        if (ctx.groupIdx < rows * heads * qTiles) {
            int[] ks = ctx.allocateIntLocalArray(MMA_KEYS * MMA_K_PITCH);
            int[] vs = ctx.allocateIntLocalArray(MMA_HEAD_DIM * MMA_V_PITCH);
            int g = ctx.groupIdx;
            int rh = g / qTiles;
            int qt = g - rh * qTiles;
            int r = rh / heads;
            int hh = rh - r * heads;
            int tid = ctx.localIdx;
            int lane = tid & 31;
            int quad = lane & 3;
            int ra = qt * MMA_QUERIES + (tid >> 5) * 16 + (lane >> 2);
            int rb = ra + 8;
            int c = quad * 2;
            int row0 = r * n;
            int qBase = row0 * 3 * d + hh * MMA_HEAD_DIM;
            int qa = qBase + ra * 3 * d;
            int qb = qBase + rb * 3 * d;
            boolean okA = ra < n;
            boolean okB = rb < n;
            int vA = 0;
            int vB = 0;
            if (okA) {
                vA = valid.get(row0 + ra);
            }
            if (okB) {
                vB = valid.get(row0 + rb);
            }
            int bidir = future.get(r);
            int lastQ = TornadoMath.min(qt * MMA_QUERIES + MMA_QUERIES, n) - 1;
            int kEnd = n;
            if (bidir == 0) {
                kEnd = lastQ + 1;
            }
            int q00 = okA ? pair(qkv, qa + 0 + c) : 0;
            int q01 = okB ? pair(qkv, qb + 0 + c) : 0;
            int q02 = okA ? pair(qkv, qa + 8 + c) : 0;
            int q03 = okB ? pair(qkv, qb + 8 + c) : 0;
            int q10 = okA ? pair(qkv, qa + 16 + c) : 0;
            int q11 = okB ? pair(qkv, qb + 16 + c) : 0;
            int q12 = okA ? pair(qkv, qa + 24 + c) : 0;
            int q13 = okB ? pair(qkv, qb + 24 + c) : 0;
            int q20 = okA ? pair(qkv, qa + 32 + c) : 0;
            int q21 = okB ? pair(qkv, qb + 32 + c) : 0;
            int q22 = okA ? pair(qkv, qa + 40 + c) : 0;
            int q23 = okB ? pair(qkv, qb + 40 + c) : 0;
            int q30 = okA ? pair(qkv, qa + 48 + c) : 0;
            int q31 = okB ? pair(qkv, qb + 48 + c) : 0;
            int q32 = okA ? pair(qkv, qa + 56 + c) : 0;
            int q33 = okB ? pair(qkv, qb + 56 + c) : 0;
            int q40 = okA ? pair(qkv, qa + 64 + c) : 0;
            int q41 = okB ? pair(qkv, qb + 64 + c) : 0;
            int q42 = okA ? pair(qkv, qa + 72 + c) : 0;
            int q43 = okB ? pair(qkv, qb + 72 + c) : 0;
            int q50 = okA ? pair(qkv, qa + 80 + c) : 0;
            int q51 = okB ? pair(qkv, qb + 80 + c) : 0;
            int q52 = okA ? pair(qkv, qa + 88 + c) : 0;
            int q53 = okB ? pair(qkv, qb + 88 + c) : 0;
            int q60 = okA ? pair(qkv, qa + 96 + c) : 0;
            int q61 = okB ? pair(qkv, qb + 96 + c) : 0;
            int q62 = okA ? pair(qkv, qa + 104 + c) : 0;
            int q63 = okB ? pair(qkv, qb + 104 + c) : 0;
            int q70 = okA ? pair(qkv, qa + 112 + c) : 0;
            int q71 = okB ? pair(qkv, qb + 112 + c) : 0;
            int q72 = okA ? pair(qkv, qa + 120 + c) : 0;
            int q73 = okB ? pair(qkv, qb + 120 + c) : 0;
            float mA = MASKED;
            float mB = MASKED;
            float lA = 0f;
            float lB = 0f;
            float o0_0 = 0f, o0_1 = 0f, o0_2 = 0f, o0_3 = 0f;
            float o1_0 = 0f, o1_1 = 0f, o1_2 = 0f, o1_3 = 0f;
            float o2_0 = 0f, o2_1 = 0f, o2_2 = 0f, o2_3 = 0f;
            float o3_0 = 0f, o3_1 = 0f, o3_2 = 0f, o3_3 = 0f;
            float o4_0 = 0f, o4_1 = 0f, o4_2 = 0f, o4_3 = 0f;
            float o5_0 = 0f, o5_1 = 0f, o5_2 = 0f, o5_3 = 0f;
            float o6_0 = 0f, o6_1 = 0f, o6_2 = 0f, o6_3 = 0f;
            float o7_0 = 0f, o7_1 = 0f, o7_2 = 0f, o7_3 = 0f;
            float o8_0 = 0f, o8_1 = 0f, o8_2 = 0f, o8_3 = 0f;
            float o9_0 = 0f, o9_1 = 0f, o9_2 = 0f, o9_3 = 0f;
            float o10_0 = 0f, o10_1 = 0f, o10_2 = 0f, o10_3 = 0f;
            float o11_0 = 0f, o11_1 = 0f, o11_2 = 0f, o11_3 = 0f;
            float o12_0 = 0f, o12_1 = 0f, o12_2 = 0f, o12_3 = 0f;
            float o13_0 = 0f, o13_1 = 0f, o13_2 = 0f, o13_3 = 0f;
            float o14_0 = 0f, o14_1 = 0f, o14_2 = 0f, o14_3 = 0f;
            float o15_0 = 0f, o15_1 = 0f, o15_2 = 0f, o15_3 = 0f;
            for (int k0 = 0; k0 < kEnd; k0 += MMA_KEYS) {
                // K: key j, dimensions 2p, 2p + 1 at ks[j·pitch + p]; V transposed: dimension e, keys 2p, 2p + 1 at vs[e·pitch + p]
                for (int i = tid; i < MMA_KEYS * MMA_HEAD_DIM / 2; i += MMA_THREADS) {
                    int j = i >> 6;
                    int p = i & 63;
                    int ko = qBase + (k0 + j) * 3 * d + d + 2 * p;
                    boolean in = k0 + j < n;
                    ks[j * MMA_K_PITCH + p] = in ? pair(qkv, ko) : 0;
                }
                for (int i = tid; i < MMA_KEYS * MMA_HEAD_DIM / 2; i += MMA_THREADS) {
                    int e = i & (MMA_HEAD_DIM - 1);
                    int p = i >> 7;
                    int vo = qBase + (k0 + 2 * p) * 3 * d + 2 * d + e;
                    int v0 = k0 + 2 * p < n ? bits(qkv, vo) : 0;
                    int v1 = k0 + 2 * p + 1 < n ? bits(qkv, vo + 3 * d) : 0;
                    vs[e * MMA_V_PITCH + p] = v0 | (v1 << 16);
                }
                ctx.localBarrier();
                HalfFloat[] fq0 = ctx.mmaFragmentA(q00, q01, q02, q03);
                HalfFloat[] fq1 = ctx.mmaFragmentA(q10, q11, q12, q13);
                HalfFloat[] fq2 = ctx.mmaFragmentA(q20, q21, q22, q23);
                HalfFloat[] fq3 = ctx.mmaFragmentA(q30, q31, q32, q33);
                HalfFloat[] fq4 = ctx.mmaFragmentA(q40, q41, q42, q43);
                HalfFloat[] fq5 = ctx.mmaFragmentA(q50, q51, q52, q53);
                HalfFloat[] fq6 = ctx.mmaFragmentA(q60, q61, q62, q63);
                HalfFloat[] fq7 = ctx.mmaFragmentA(q70, q71, q72, q73);
                float[] s0 = ctx.mmaFragment(0f);
                int kr0 = (0 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk0_0 = ctx.mmaFragmentB(ks[kr0 + 0], ks[kr0 + 4]);
                s0 = ctx.mma(fq0, fk0_0, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_1 = ctx.mmaFragmentB(ks[kr0 + 8], ks[kr0 + 12]);
                s0 = ctx.mma(fq1, fk0_1, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_2 = ctx.mmaFragmentB(ks[kr0 + 16], ks[kr0 + 20]);
                s0 = ctx.mma(fq2, fk0_2, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_3 = ctx.mmaFragmentB(ks[kr0 + 24], ks[kr0 + 28]);
                s0 = ctx.mma(fq3, fk0_3, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_4 = ctx.mmaFragmentB(ks[kr0 + 32], ks[kr0 + 36]);
                s0 = ctx.mma(fq4, fk0_4, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_5 = ctx.mmaFragmentB(ks[kr0 + 40], ks[kr0 + 44]);
                s0 = ctx.mma(fq5, fk0_5, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_6 = ctx.mmaFragmentB(ks[kr0 + 48], ks[kr0 + 52]);
                s0 = ctx.mma(fq6, fk0_6, s0, MMAShape.M16N8K16);
                HalfFloat[] fk0_7 = ctx.mmaFragmentB(ks[kr0 + 56], ks[kr0 + 60]);
                s0 = ctx.mma(fq7, fk0_7, s0, MMAShape.M16N8K16);
                float[] s1 = ctx.mmaFragment(0f);
                int kr1 = (8 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk1_0 = ctx.mmaFragmentB(ks[kr1 + 0], ks[kr1 + 4]);
                s1 = ctx.mma(fq0, fk1_0, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_1 = ctx.mmaFragmentB(ks[kr1 + 8], ks[kr1 + 12]);
                s1 = ctx.mma(fq1, fk1_1, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_2 = ctx.mmaFragmentB(ks[kr1 + 16], ks[kr1 + 20]);
                s1 = ctx.mma(fq2, fk1_2, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_3 = ctx.mmaFragmentB(ks[kr1 + 24], ks[kr1 + 28]);
                s1 = ctx.mma(fq3, fk1_3, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_4 = ctx.mmaFragmentB(ks[kr1 + 32], ks[kr1 + 36]);
                s1 = ctx.mma(fq4, fk1_4, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_5 = ctx.mmaFragmentB(ks[kr1 + 40], ks[kr1 + 44]);
                s1 = ctx.mma(fq5, fk1_5, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_6 = ctx.mmaFragmentB(ks[kr1 + 48], ks[kr1 + 52]);
                s1 = ctx.mma(fq6, fk1_6, s1, MMAShape.M16N8K16);
                HalfFloat[] fk1_7 = ctx.mmaFragmentB(ks[kr1 + 56], ks[kr1 + 60]);
                s1 = ctx.mma(fq7, fk1_7, s1, MMAShape.M16N8K16);
                float[] s2 = ctx.mmaFragment(0f);
                int kr2 = (16 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk2_0 = ctx.mmaFragmentB(ks[kr2 + 0], ks[kr2 + 4]);
                s2 = ctx.mma(fq0, fk2_0, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_1 = ctx.mmaFragmentB(ks[kr2 + 8], ks[kr2 + 12]);
                s2 = ctx.mma(fq1, fk2_1, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_2 = ctx.mmaFragmentB(ks[kr2 + 16], ks[kr2 + 20]);
                s2 = ctx.mma(fq2, fk2_2, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_3 = ctx.mmaFragmentB(ks[kr2 + 24], ks[kr2 + 28]);
                s2 = ctx.mma(fq3, fk2_3, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_4 = ctx.mmaFragmentB(ks[kr2 + 32], ks[kr2 + 36]);
                s2 = ctx.mma(fq4, fk2_4, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_5 = ctx.mmaFragmentB(ks[kr2 + 40], ks[kr2 + 44]);
                s2 = ctx.mma(fq5, fk2_5, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_6 = ctx.mmaFragmentB(ks[kr2 + 48], ks[kr2 + 52]);
                s2 = ctx.mma(fq6, fk2_6, s2, MMAShape.M16N8K16);
                HalfFloat[] fk2_7 = ctx.mmaFragmentB(ks[kr2 + 56], ks[kr2 + 60]);
                s2 = ctx.mma(fq7, fk2_7, s2, MMAShape.M16N8K16);
                float[] s3 = ctx.mmaFragment(0f);
                int kr3 = (24 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk3_0 = ctx.mmaFragmentB(ks[kr3 + 0], ks[kr3 + 4]);
                s3 = ctx.mma(fq0, fk3_0, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_1 = ctx.mmaFragmentB(ks[kr3 + 8], ks[kr3 + 12]);
                s3 = ctx.mma(fq1, fk3_1, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_2 = ctx.mmaFragmentB(ks[kr3 + 16], ks[kr3 + 20]);
                s3 = ctx.mma(fq2, fk3_2, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_3 = ctx.mmaFragmentB(ks[kr3 + 24], ks[kr3 + 28]);
                s3 = ctx.mma(fq3, fk3_3, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_4 = ctx.mmaFragmentB(ks[kr3 + 32], ks[kr3 + 36]);
                s3 = ctx.mma(fq4, fk3_4, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_5 = ctx.mmaFragmentB(ks[kr3 + 40], ks[kr3 + 44]);
                s3 = ctx.mma(fq5, fk3_5, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_6 = ctx.mmaFragmentB(ks[kr3 + 48], ks[kr3 + 52]);
                s3 = ctx.mma(fq6, fk3_6, s3, MMAShape.M16N8K16);
                HalfFloat[] fk3_7 = ctx.mmaFragmentB(ks[kr3 + 56], ks[kr3 + 60]);
                s3 = ctx.mma(fq7, fk3_7, s3, MMAShape.M16N8K16);
                int kv00 = keyValid(valid, row0, n, k0 + 0 + c);
                float x00 = allowed(ra, k0 + 0 + c, vA, kv00, bidir) != 0 ? s0[0] * scale : MASKED;
                float y00 = allowed(rb, k0 + 0 + c, vB, kv00, bidir) != 0 ? s0[2] * scale : MASKED;
                int kv01 = keyValid(valid, row0, n, k0 + 1 + c);
                float x01 = allowed(ra, k0 + 1 + c, vA, kv01, bidir) != 0 ? s0[1] * scale : MASKED;
                float y01 = allowed(rb, k0 + 1 + c, vB, kv01, bidir) != 0 ? s0[3] * scale : MASKED;
                int kv10 = keyValid(valid, row0, n, k0 + 8 + c);
                float x10 = allowed(ra, k0 + 8 + c, vA, kv10, bidir) != 0 ? s1[0] * scale : MASKED;
                float y10 = allowed(rb, k0 + 8 + c, vB, kv10, bidir) != 0 ? s1[2] * scale : MASKED;
                int kv11 = keyValid(valid, row0, n, k0 + 9 + c);
                float x11 = allowed(ra, k0 + 9 + c, vA, kv11, bidir) != 0 ? s1[1] * scale : MASKED;
                float y11 = allowed(rb, k0 + 9 + c, vB, kv11, bidir) != 0 ? s1[3] * scale : MASKED;
                int kv20 = keyValid(valid, row0, n, k0 + 16 + c);
                float x20 = allowed(ra, k0 + 16 + c, vA, kv20, bidir) != 0 ? s2[0] * scale : MASKED;
                float y20 = allowed(rb, k0 + 16 + c, vB, kv20, bidir) != 0 ? s2[2] * scale : MASKED;
                int kv21 = keyValid(valid, row0, n, k0 + 17 + c);
                float x21 = allowed(ra, k0 + 17 + c, vA, kv21, bidir) != 0 ? s2[1] * scale : MASKED;
                float y21 = allowed(rb, k0 + 17 + c, vB, kv21, bidir) != 0 ? s2[3] * scale : MASKED;
                int kv30 = keyValid(valid, row0, n, k0 + 24 + c);
                float x30 = allowed(ra, k0 + 24 + c, vA, kv30, bidir) != 0 ? s3[0] * scale : MASKED;
                float y30 = allowed(rb, k0 + 24 + c, vB, kv30, bidir) != 0 ? s3[2] * scale : MASKED;
                int kv31 = keyValid(valid, row0, n, k0 + 25 + c);
                float x31 = allowed(ra, k0 + 25 + c, vA, kv31, bidir) != 0 ? s3[1] * scale : MASKED;
                float y31 = allowed(rb, k0 + 25 + c, vB, kv31, bidir) != 0 ? s3[3] * scale : MASKED;
                float bA = TornadoMath.max(TornadoMath.max(TornadoMath.max(x00, x01), TornadoMath.max(x10, x11)), TornadoMath.max(TornadoMath.max(x20, x21), TornadoMath.max(x30, x31)));
                float bB = TornadoMath.max(TornadoMath.max(TornadoMath.max(y00, y01), TornadoMath.max(y10, y11)), TornadoMath.max(TornadoMath.max(y20, y21), TornadoMath.max(y30, y31)));
                bA = TornadoMath.max(bA, ctx.simdShuffle(bA, lane ^ 1));
                bA = TornadoMath.max(bA, ctx.simdShuffle(bA, lane ^ 2));
                bB = TornadoMath.max(bB, ctx.simdShuffle(bB, lane ^ 1));
                bB = TornadoMath.max(bB, ctx.simdShuffle(bB, lane ^ 2));
                float nA = TornadoMath.max(mA, bA);
                float nB = TornadoMath.max(mB, bB);
                float cA = bA > VALID ? TornadoMath.exp(mA - nA) : 1f;
                float cB = bB > VALID ? TornadoMath.exp(mB - nB) : 1f;
                float pa00 = x00 > VALID ? TornadoMath.exp(x00 - nA) : 0f;
                float pb00 = y00 > VALID ? TornadoMath.exp(y00 - nB) : 0f;
                float pa01 = x01 > VALID ? TornadoMath.exp(x01 - nA) : 0f;
                float pb01 = y01 > VALID ? TornadoMath.exp(y01 - nB) : 0f;
                float pa10 = x10 > VALID ? TornadoMath.exp(x10 - nA) : 0f;
                float pb10 = y10 > VALID ? TornadoMath.exp(y10 - nB) : 0f;
                float pa11 = x11 > VALID ? TornadoMath.exp(x11 - nA) : 0f;
                float pb11 = y11 > VALID ? TornadoMath.exp(y11 - nB) : 0f;
                float pa20 = x20 > VALID ? TornadoMath.exp(x20 - nA) : 0f;
                float pb20 = y20 > VALID ? TornadoMath.exp(y20 - nB) : 0f;
                float pa21 = x21 > VALID ? TornadoMath.exp(x21 - nA) : 0f;
                float pb21 = y21 > VALID ? TornadoMath.exp(y21 - nB) : 0f;
                float pa30 = x30 > VALID ? TornadoMath.exp(x30 - nA) : 0f;
                float pb30 = y30 > VALID ? TornadoMath.exp(y30 - nB) : 0f;
                float pa31 = x31 > VALID ? TornadoMath.exp(x31 - nA) : 0f;
                float pb31 = y31 > VALID ? TornadoMath.exp(y31 - nB) : 0f;
                lA = lA * cA + pa00 + pa01 + pa10 + pa11 + pa20 + pa21 + pa30 + pa31;
                lB = lB * cB + pb00 + pb01 + pb10 + pb11 + pb20 + pb21 + pb30 + pb31;
                mA = nA;
                mB = nB;
                HalfFloat[] fp0 = ctx.mmaFragmentA(pack(pa00, pa01), pack(pb00, pb01), pack(pa10, pa11), pack(pb10, pb11));
                HalfFloat[] fp1 = ctx.mmaFragmentA(pack(pa20, pa21), pack(pb20, pb21), pack(pa30, pa31), pack(pb30, pb31));
                int vr0 = (0 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t0 = ctx.mmaFragment(0f);
                t0 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr0 + 0], vs[vr0 + 4]), t0, MMAShape.M16N8K16);
                t0 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr0 + 8], vs[vr0 + 12]), t0, MMAShape.M16N8K16);
                o0_0 = o0_0 * cA + t0[0];
                o0_1 = o0_1 * cA + t0[1];
                o0_2 = o0_2 * cB + t0[2];
                o0_3 = o0_3 * cB + t0[3];
                int vr1 = (8 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t1 = ctx.mmaFragment(0f);
                t1 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr1 + 0], vs[vr1 + 4]), t1, MMAShape.M16N8K16);
                t1 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr1 + 8], vs[vr1 + 12]), t1, MMAShape.M16N8K16);
                o1_0 = o1_0 * cA + t1[0];
                o1_1 = o1_1 * cA + t1[1];
                o1_2 = o1_2 * cB + t1[2];
                o1_3 = o1_3 * cB + t1[3];
                int vr2 = (16 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t2 = ctx.mmaFragment(0f);
                t2 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr2 + 0], vs[vr2 + 4]), t2, MMAShape.M16N8K16);
                t2 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr2 + 8], vs[vr2 + 12]), t2, MMAShape.M16N8K16);
                o2_0 = o2_0 * cA + t2[0];
                o2_1 = o2_1 * cA + t2[1];
                o2_2 = o2_2 * cB + t2[2];
                o2_3 = o2_3 * cB + t2[3];
                int vr3 = (24 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t3 = ctx.mmaFragment(0f);
                t3 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr3 + 0], vs[vr3 + 4]), t3, MMAShape.M16N8K16);
                t3 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr3 + 8], vs[vr3 + 12]), t3, MMAShape.M16N8K16);
                o3_0 = o3_0 * cA + t3[0];
                o3_1 = o3_1 * cA + t3[1];
                o3_2 = o3_2 * cB + t3[2];
                o3_3 = o3_3 * cB + t3[3];
                int vr4 = (32 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t4 = ctx.mmaFragment(0f);
                t4 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr4 + 0], vs[vr4 + 4]), t4, MMAShape.M16N8K16);
                t4 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr4 + 8], vs[vr4 + 12]), t4, MMAShape.M16N8K16);
                o4_0 = o4_0 * cA + t4[0];
                o4_1 = o4_1 * cA + t4[1];
                o4_2 = o4_2 * cB + t4[2];
                o4_3 = o4_3 * cB + t4[3];
                int vr5 = (40 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t5 = ctx.mmaFragment(0f);
                t5 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr5 + 0], vs[vr5 + 4]), t5, MMAShape.M16N8K16);
                t5 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr5 + 8], vs[vr5 + 12]), t5, MMAShape.M16N8K16);
                o5_0 = o5_0 * cA + t5[0];
                o5_1 = o5_1 * cA + t5[1];
                o5_2 = o5_2 * cB + t5[2];
                o5_3 = o5_3 * cB + t5[3];
                int vr6 = (48 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t6 = ctx.mmaFragment(0f);
                t6 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr6 + 0], vs[vr6 + 4]), t6, MMAShape.M16N8K16);
                t6 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr6 + 8], vs[vr6 + 12]), t6, MMAShape.M16N8K16);
                o6_0 = o6_0 * cA + t6[0];
                o6_1 = o6_1 * cA + t6[1];
                o6_2 = o6_2 * cB + t6[2];
                o6_3 = o6_3 * cB + t6[3];
                int vr7 = (56 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t7 = ctx.mmaFragment(0f);
                t7 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr7 + 0], vs[vr7 + 4]), t7, MMAShape.M16N8K16);
                t7 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr7 + 8], vs[vr7 + 12]), t7, MMAShape.M16N8K16);
                o7_0 = o7_0 * cA + t7[0];
                o7_1 = o7_1 * cA + t7[1];
                o7_2 = o7_2 * cB + t7[2];
                o7_3 = o7_3 * cB + t7[3];
                int vr8 = (64 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t8 = ctx.mmaFragment(0f);
                t8 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr8 + 0], vs[vr8 + 4]), t8, MMAShape.M16N8K16);
                t8 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr8 + 8], vs[vr8 + 12]), t8, MMAShape.M16N8K16);
                o8_0 = o8_0 * cA + t8[0];
                o8_1 = o8_1 * cA + t8[1];
                o8_2 = o8_2 * cB + t8[2];
                o8_3 = o8_3 * cB + t8[3];
                int vr9 = (72 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t9 = ctx.mmaFragment(0f);
                t9 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr9 + 0], vs[vr9 + 4]), t9, MMAShape.M16N8K16);
                t9 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr9 + 8], vs[vr9 + 12]), t9, MMAShape.M16N8K16);
                o9_0 = o9_0 * cA + t9[0];
                o9_1 = o9_1 * cA + t9[1];
                o9_2 = o9_2 * cB + t9[2];
                o9_3 = o9_3 * cB + t9[3];
                int vr10 = (80 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t10 = ctx.mmaFragment(0f);
                t10 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr10 + 0], vs[vr10 + 4]), t10, MMAShape.M16N8K16);
                t10 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr10 + 8], vs[vr10 + 12]), t10, MMAShape.M16N8K16);
                o10_0 = o10_0 * cA + t10[0];
                o10_1 = o10_1 * cA + t10[1];
                o10_2 = o10_2 * cB + t10[2];
                o10_3 = o10_3 * cB + t10[3];
                int vr11 = (88 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t11 = ctx.mmaFragment(0f);
                t11 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr11 + 0], vs[vr11 + 4]), t11, MMAShape.M16N8K16);
                t11 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr11 + 8], vs[vr11 + 12]), t11, MMAShape.M16N8K16);
                o11_0 = o11_0 * cA + t11[0];
                o11_1 = o11_1 * cA + t11[1];
                o11_2 = o11_2 * cB + t11[2];
                o11_3 = o11_3 * cB + t11[3];
                int vr12 = (96 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t12 = ctx.mmaFragment(0f);
                t12 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr12 + 0], vs[vr12 + 4]), t12, MMAShape.M16N8K16);
                t12 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr12 + 8], vs[vr12 + 12]), t12, MMAShape.M16N8K16);
                o12_0 = o12_0 * cA + t12[0];
                o12_1 = o12_1 * cA + t12[1];
                o12_2 = o12_2 * cB + t12[2];
                o12_3 = o12_3 * cB + t12[3];
                int vr13 = (104 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t13 = ctx.mmaFragment(0f);
                t13 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr13 + 0], vs[vr13 + 4]), t13, MMAShape.M16N8K16);
                t13 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr13 + 8], vs[vr13 + 12]), t13, MMAShape.M16N8K16);
                o13_0 = o13_0 * cA + t13[0];
                o13_1 = o13_1 * cA + t13[1];
                o13_2 = o13_2 * cB + t13[2];
                o13_3 = o13_3 * cB + t13[3];
                int vr14 = (112 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t14 = ctx.mmaFragment(0f);
                t14 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr14 + 0], vs[vr14 + 4]), t14, MMAShape.M16N8K16);
                t14 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr14 + 8], vs[vr14 + 12]), t14, MMAShape.M16N8K16);
                o14_0 = o14_0 * cA + t14[0];
                o14_1 = o14_1 * cA + t14[1];
                o14_2 = o14_2 * cB + t14[2];
                o14_3 = o14_3 * cB + t14[3];
                int vr15 = (120 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t15 = ctx.mmaFragment(0f);
                t15 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr15 + 0], vs[vr15 + 4]), t15, MMAShape.M16N8K16);
                t15 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr15 + 8], vs[vr15 + 12]), t15, MMAShape.M16N8K16);
                o15_0 = o15_0 * cA + t15[0];
                o15_1 = o15_1 * cA + t15[1];
                o15_2 = o15_2 * cB + t15[2];
                o15_3 = o15_3 * cB + t15[3];
                ctx.localBarrier();
            }
            lA = lA + ctx.simdShuffle(lA, lane ^ 1);
            lA = lA + ctx.simdShuffle(lA, lane ^ 2);
            lB = lB + ctx.simdShuffle(lB, lane ^ 1);
            lB = lB + ctx.simdShuffle(lB, lane ^ 2);
            // an all-padding query reads no key and gets zeros
            float iA = lA > 0f ? 1f / lA : 0f;
            float iB = lB > 0f ? 1f / lB : 0f;
            int oa = (row0 + ra) * d + hh * MMA_HEAD_DIM + c;
            int ob = (row0 + rb) * d + hh * MMA_HEAD_DIM + c;
            if (okA) {
                store(out, th, twin, oa + 0, o0_0 * iA);
                store(out, th, twin, oa + 1, o0_1 * iA);
                store(out, th, twin, oa + 8, o1_0 * iA);
                store(out, th, twin, oa + 9, o1_1 * iA);
                store(out, th, twin, oa + 16, o2_0 * iA);
                store(out, th, twin, oa + 17, o2_1 * iA);
                store(out, th, twin, oa + 24, o3_0 * iA);
                store(out, th, twin, oa + 25, o3_1 * iA);
                store(out, th, twin, oa + 32, o4_0 * iA);
                store(out, th, twin, oa + 33, o4_1 * iA);
                store(out, th, twin, oa + 40, o5_0 * iA);
                store(out, th, twin, oa + 41, o5_1 * iA);
                store(out, th, twin, oa + 48, o6_0 * iA);
                store(out, th, twin, oa + 49, o6_1 * iA);
                store(out, th, twin, oa + 56, o7_0 * iA);
                store(out, th, twin, oa + 57, o7_1 * iA);
                store(out, th, twin, oa + 64, o8_0 * iA);
                store(out, th, twin, oa + 65, o8_1 * iA);
                store(out, th, twin, oa + 72, o9_0 * iA);
                store(out, th, twin, oa + 73, o9_1 * iA);
                store(out, th, twin, oa + 80, o10_0 * iA);
                store(out, th, twin, oa + 81, o10_1 * iA);
                store(out, th, twin, oa + 88, o11_0 * iA);
                store(out, th, twin, oa + 89, o11_1 * iA);
                store(out, th, twin, oa + 96, o12_0 * iA);
                store(out, th, twin, oa + 97, o12_1 * iA);
                store(out, th, twin, oa + 104, o13_0 * iA);
                store(out, th, twin, oa + 105, o13_1 * iA);
                store(out, th, twin, oa + 112, o14_0 * iA);
                store(out, th, twin, oa + 113, o14_1 * iA);
                store(out, th, twin, oa + 120, o15_0 * iA);
                store(out, th, twin, oa + 121, o15_1 * iA);
            }
            if (okB) {
                store(out, th, twin, ob + 0, o0_2 * iB);
                store(out, th, twin, ob + 1, o0_3 * iB);
                store(out, th, twin, ob + 8, o1_2 * iB);
                store(out, th, twin, ob + 9, o1_3 * iB);
                store(out, th, twin, ob + 16, o2_2 * iB);
                store(out, th, twin, ob + 17, o2_3 * iB);
                store(out, th, twin, ob + 24, o3_2 * iB);
                store(out, th, twin, ob + 25, o3_3 * iB);
                store(out, th, twin, ob + 32, o4_2 * iB);
                store(out, th, twin, ob + 33, o4_3 * iB);
                store(out, th, twin, ob + 40, o5_2 * iB);
                store(out, th, twin, ob + 41, o5_3 * iB);
                store(out, th, twin, ob + 48, o6_2 * iB);
                store(out, th, twin, ob + 49, o6_3 * iB);
                store(out, th, twin, ob + 56, o7_2 * iB);
                store(out, th, twin, ob + 57, o7_3 * iB);
                store(out, th, twin, ob + 64, o8_2 * iB);
                store(out, th, twin, ob + 65, o8_3 * iB);
                store(out, th, twin, ob + 72, o9_2 * iB);
                store(out, th, twin, ob + 73, o9_3 * iB);
                store(out, th, twin, ob + 80, o10_2 * iB);
                store(out, th, twin, ob + 81, o10_3 * iB);
                store(out, th, twin, ob + 88, o11_2 * iB);
                store(out, th, twin, ob + 89, o11_3 * iB);
                store(out, th, twin, ob + 96, o12_2 * iB);
                store(out, th, twin, ob + 97, o12_3 * iB);
                store(out, th, twin, ob + 104, o13_2 * iB);
                store(out, th, twin, ob + 105, o13_3 * iB);
                store(out, th, twin, ob + 112, o14_2 * iB);
                store(out, th, twin, ob + 113, o14_3 * iB);
                store(out, th, twin, ob + 120, o15_2 * iB);
                store(out, th, twin, ob + 121, o15_3 * iB);
            }
        }
    }

    /** Whether a query at {@code qi} reads the key at {@code k}, as 0 or 1 (int masks, not short-circuit conditions). */
    static int allowed(int qi, int k, int qValid, int kValid, int bidir) {
        int causal = k <= qi ? 1 : 0;
        return qValid * kValid * TornadoMath.max(causal, bidir);
    }

    /** {@code valid[row0 + k]} for a key inside the row, 0 past it. */
    static int keyValid(IntArray valid, int row0, int n, int k) {
        int v = 0;
        if (k < n) {
            v = valid.get(row0 + k);
        }
        return v;
    }

    /** {@code out[i] = v}, and its FP16 copy, saturated to ±{@code HALF_MAX}, when {@code twin} is 1. */
    static void store(FloatArray out, HalfFloatArray th, int twin, int i, float v) {
        out.set(i, v);
        if (twin == 1) {
            th.set(i, new HalfFloat(TornadoMath.clamp(v, -Kernels.HALF_MAX, Kernels.HALF_MAX)));
        }
    }

    /** {@code x} and {@code y} as FP16 in one register, {@code x} in bits 0-15: a half pair of an {@code mma} fragment. */
    static int pack(float x, float y) {
        return (new HalfFloat(x).getHalfFloatValue() & 0xFFFF) | ((new HalfFloat(y).getHalfFloatValue() & 0xFFFF) << 16);
    }

    /** The FP16 bits of {@code a[i]}, in the low 16 bits. */
    static int bits(HalfFloatArray a, int i) {
        return a.get(i).getHalfFloatValue() & 0xFFFF;
    }

    /** {@code a[i]} and {@code a[i + 1]} as one register, {@code a[i]} in bits 0-15: a half pair of an {@code mma} fragment. */
    static int pair(HalfFloatArray a, int i) {
        return bits(a, i) | (bits(a, i + 1) << 16);
    }

    /** {@code a[i] = v} in FP16, saturated to ±{@code HALF_MAX}. */
    static void setHalf(HalfFloatArray a, int i, float v) {
        a.set(i, new HalfFloat(TornadoMath.clamp(v, -Kernels.HALF_MAX, Kernels.HALF_MAX)));
    }

    /** What FP16 rounding loses of {@code x}: {@code x = half(x) + lo(x)} to about 2⁻²². */
    static float lo(float x) {
        return x - new HalfFloat(x).getFloat32();
    }

    /**
     * Attention across the rows of each series at each patch, per (row, patch, head): a valid query attends to
     * the valid patches of its series' rows at the same position; a known-future covariate reads only
     * known-future covariates. One work-group of {@link #GROUP_THREADS}, one thread per head dimension, walks
     * the series' rows ({@code groupRows[groupStart[g] …]}) with an online softmax, each dot product reduced
     * in local memory. Also writes the FP16 copy {@code th} of {@code out} when {@code twin} is 1.
     */
    public static void groupAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray valid, IntArray future,
            IntArray groupOf, IntArray groupStart, IntArray groupRows, int rows, int n, int heads, int hd, int d, float scale,
            HalfFloatArray th, int twin) {
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
                store(out, th, twin, tok * d + head * hd + tid, den > 0f ? acc / den : 0f);
            }
        }
    }

    /** {@link #groupAttention} on {@code q|k|v} projections held in FP16. */
    public static void groupAttentionHalf(KernelContext ctx, HalfFloatArray qkv, FloatArray out, IntArray valid, IntArray future,
            IntArray groupOf, IntArray groupStart, IntArray groupRows, int rows, int n, int heads, int hd, int d, float scale,
            HalfFloatArray th, int twin) {
        if (ctx.groupIdx < rows * n * heads) {
            int head = ctx.groupIdx % heads;
            int tok = ctx.groupIdx / heads;
            int a = tok / n;
            int s = tok - a * n;
            int tid = ctx.localIdx;
            float[] part = ctx.allocateFloatLocalArray(GROUP_THREADS);
            boolean active = tid < hd;
            float q = active ? qkv.get(tok * 3 * d + head * hd + tid).getFloat32() : 0f;
            float mx = -3.0e38f;
            float den = 0f;
            float acc = 0f;
            int g = groupOf.get(a);
            int start = groupStart.get(g);
            int end = groupStart.get(g + 1);
            boolean query = valid.get(tok) == 1;
            boolean fa = future.get(a) == 1;
            if (end - start == 1) {
                // a series of one row: a valid query reads only itself, with weight 1
                den = query ? 1f : 0f;
                acc = query && active ? qkv.get(tok * 3 * d + 2 * d + head * hd + tid).getFloat32() : 0f;
                end = start;
            }
            for (int m = start; m < end; m++) {
                int b = groupRows.get(m);
                int kt = b * n + s;
                boolean allowed = query && valid.get(kt) == 1 && !(fa && future.get(b) == 0);
                if (allowed) {
                    part[tid] = active ? q * qkv.get(kt * 3 * d + d + head * hd + tid).getFloat32() : 0f;
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
                    float v = active ? qkv.get(kt * 3 * d + 2 * d + head * hd + tid).getFloat32() : 0f;
                    acc = acc * al + pr * v;
                    mx = nm;
                }
            }
            if (active) {
                store(out, th, twin, tok * d + head * hd + tid, den > 0f ? acc / den : 0f);
            }
        }
    }

    /**
     * {@code x[r] += b} when {@code bOff >= 0} (the bias of the GEMM that accumulated into the residual stream
     * without one), then {@code y[r] = RMSNorm(x[r])·w}: one work-group of {@link #NORM_THREADS} per row. Also
     * writes the FP16 copy {@code th} of {@code y} when {@code twin} is 1.
     */
    public static void biasRmsNorm(KernelContext ctx, FloatArray x, FloatArray y, FloatArray w, int bOff, int wOff, int d,
            float eps, int rows, HalfFloatArray th, int twin) {
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
                store(y, th, twin, base + i, x.get(base + i) * inv * w.get(wOff + i));
            }
        }
    }

    /**
     * {@code out[r, j] = SiLU(g + b[j]) · (u + b[f + j])} on the gate|value projection {@code [n, 2f]} that
     * its GEMM wrote without the bias, and its FP16 copy {@code th} when {@code twin} is 1.
     */
    public static void biasSwiGlu(KernelContext ctx, FloatArray big, FloatArray out, FloatArray w, int bOff, int n, int ffn,
            HalfFloatArray th, int twin) {
        int id = ctx.globalIdx;
        if (id < n * ffn) {
            int r = id / ffn;
            int j = id - r * ffn;
            float g = big.get(r * 2 * ffn + j) + w.get(bOff + j);
            float u = big.get(r * 2 * ffn + ffn + j) + w.get(bOff + ffn + j);
            store(out, th, twin, id, g / (1f + TornadoMath.exp(-g)) * u);
        }
    }

    /** {@link #biasSwiGlu} on a gate|value projection its GEMM wrote in FP16. */
    public static void biasSwiGluHalf(KernelContext ctx, HalfFloatArray big, FloatArray out, FloatArray w, int bOff, int n, int ffn,
            HalfFloatArray th, int twin) {
        int id = ctx.globalIdx;
        if (id < n * ffn) {
            int r = id / ffn;
            int j = id - r * ffn;
            float g = big.get(r * 2 * ffn + j).getFloat32() + w.get(bOff + j);
            float u = big.get(r * 2 * ffn + ffn + j).getFloat32() + w.get(bOff + ffn + j);
            store(out, th, twin, id, g / (1f + TornadoMath.exp(-g)) * u);
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
