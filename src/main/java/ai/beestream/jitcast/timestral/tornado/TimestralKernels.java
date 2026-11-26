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
package ai.beestream.jitcast.timestral.tornado;

import ai.beestream.jitcast.tornado.Kernels;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The TornadoVM kernels specific to Timestral (portable: OpenCL, Metal, CUDA): token assembly,
 * RoPE, time and variate attention with the {@code BlockLayout} masks, the sigmoid gate and the
 * decoder-token gather. The GEMMs, RMSNorm and SwiGLU are the shared {@code tornado.Kernels}. Each
 * kernel is a device version of a {@code timestral.cpu.CpuTimestral} stage and is tested against
 * it.
 *
 * <p>Activations are row-major {@code [rows · tokens · width]}, each row's encoder tokens followed
 * by its decoder tokens; {@code q|k|v} is heads-major per token. Every kernel is a
 * {@link KernelContext} kernel launched with an explicit worker grid; element-wise ones map
 * {@code ctx.globalIdx} to one item and bounds-check it.
 */
public final class TimestralKernels {

    /** Queries per work-group of the attention kernels. */
    public static final int ATTN_QUERIES = 16;
    /** Threads per work-group of the attention kernels. */
    public static final int ATTN_THREADS = 128;
    /** Threads that share one query's dot product. */
    static final int LANES = ATTN_THREADS / ATTN_QUERIES;
    /** Threads, queries (16 per warp), keys per block and head width of {@link #timeAttentionMma}. */
    public static final int MMA_THREADS = 128;
    public static final int MMA_QUERIES = 64;
    public static final int MMA_KEYS = 32;
    public static final int MMA_HEAD_DIM = 64;
    /** Shared-memory row pitches, in FP16 pairs, of its K and transposed V tiles: padded so a fragment read hits 32 banks. */
    static final int MMA_K_PITCH = MMA_HEAD_DIM / 2 + 4;
    static final int MMA_V_PITCH = MMA_KEYS / 2 + 4;
    /** A masked score of {@link #timeAttentionMma}, and the bound above which a score is a real one. */
    static final float MASKED = -3.4e38f;
    static final float VALID = -3.0e38f;
    /** The widest attention head the kernels support. */
    public static final int MAX_HEAD_DIM = 128;

    private TimestralKernels() {}

    /**
     * {@code x [(r·n + t)·d + j]}: the patch embedding {@code emb [(r·enc + t)·d + j]} for the encoder
     * tokens {@code t < enc}, the forecast query (at {@code queryOff} in the packed weights) for the
     * decoder tokens.
     */
    public static void assembleTokens(KernelContext ctx, FloatArray emb, FloatArray w, int queryOff, FloatArray x, int rows,
            int enc, int n, int d) {
        int id = ctx.globalIdx;
        if (id < rows * n * d) {
            int j = id % d;
            int tok = id / d;
            int t = tok % n;
            int r = tok / n;
            float v;
            if (t < enc) {
                v = emb.get((r * enc + t) * d + j);
            } else {
                v = w.get(queryOff + j);
            }
            x.set(id, v);
        }
    }

    /**
     * RoPE on q and k in place: per token and head, dimensions {@code e} and {@code e + dh/2} rotate
     * together, with {@code cos, sin} read from {@code table [(pos[token])·dh + e]} and
     * {@code [.. + dh/2 + e]} (the host builds the table, so large angles are exact). One thread per
     * (token, head, {@code e < dh/2}).
     */
    public static void rope(KernelContext ctx, FloatArray qkv, IntArray pos, FloatArray table, int tokens, int heads, int dh,
            int d) {
        int half = dh / 2;
        int id = ctx.globalIdx;
        if (id < tokens * heads * half) {
            int e = id % half;
            int th = id / half;
            int hh = th % heads;
            int tok = th / heads;
            int p = pos.get(tok);
            float c = table.get(p * dh + e);
            float s = table.get(p * dh + half + e);
            int qb = tok * 3 * d + hh * dh;
            float a = qkv.get(qb + e);
            float b = qkv.get(qb + half + e);
            qkv.set(qb + e, a * c + -b * s);
            qkv.set(qb + half + e, b * c + a * s);
            int kb = qb + d;
            float ka = qkv.get(kb + e);
            float kb2 = qkv.get(kb + half + e);
            qkv.set(kb + e, ka * c + -kb2 * s);
            qkv.set(kb + half + e, kb2 * c + ka * s);
        }
    }

    /**
     * Attention along time, flash style: one work-group of {@link #ATTN_THREADS} threads per
     * (row, head, {@link #ATTN_QUERIES} consecutive query tokens). Each key's k and v are loaded once
     * into local memory for every query of the block; {@link #LANES} threads form one query's dot
     * product, its first lane applies the mask and the online softmax, and thread {@code j < dh}
     * accumulates output dimension {@code j} of every query.
     *
     * <p>Masks ({@code BlockLayout.timeAllowed}): an encoder query {@code i < enc} reads the valid
     * encoder keys {@code j ≤ i} (every valid encoder key on a known-covariate row) and itself; a
     * decoder query reads the valid context keys {@code j < ll} and the decoder tokens up to itself.
     * Writes {@code out [(r·n + t)·d + head·dh + j]}.
     */
    public static void timeAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray keyValid, IntArray known,
            int rows, int enc, int ll, int n, int heads, int dh, int d) {
        if (ctx.groupIdx < rows * heads * ((n + ATTN_QUERIES - 1) / ATTN_QUERIES)) {
            int wg = ctx.groupIdx;
            int tid = ctx.localIdx;
            int tiles = (n + ATTN_QUERIES - 1) / ATTN_QUERIES;
            int tile = wg % tiles;
            int hh = (wg / tiles) % heads;
            int r = wg / (tiles * heads);
            int q0 = tile * ATTN_QUERIES;
            int qn = TornadoMath.min(ATTN_QUERIES, n - q0);
            float scale = 1f / TornadoMath.sqrt(dh);
            float[] qs = ctx.allocateFloatLocalArray(ATTN_QUERIES * MAX_HEAD_DIM);
            float[] kv = ctx.allocateFloatLocalArray(2 * MAX_HEAD_DIM);
            float[] part = ctx.allocateFloatLocalArray(ATTN_THREADS);
            float[] mRun = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] lRun = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] corr = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] pw = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            // one scalar per query of ATTN_QUERIES = 16 (not an array), so every kernel compiler keeps them in registers
            float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f, a4 = 0f, a5 = 0f, a6 = 0f, a7 = 0f; float a8 = 0f, a9 = 0f, a10 = 0f, a11 = 0f, a12 = 0f, a13 = 0f, a14 = 0f, a15 = 0f;
            for (int qq = 0; qq < ATTN_QUERIES; qq++) {
                if (qq < qn) {
                    for (int e = tid; e < dh; e += ATTN_THREADS) {
                        qs[qq * MAX_HEAD_DIM + e] = qkv.get((r * n + q0 + qq) * 3 * d + hh * dh + e);
                    }
                }
            }
            if (tid < ATTN_QUERIES) {
                mRun[tid] = -3.4e38f;
                lRun[tid] = 0f;
            }
            int isKnown = known.get(r);
            int lastQ = q0 + qn - 1;
            int kEnd = n;
            if (lastQ < enc) {
                kEnd = isKnown == 1 ? enc : lastQ + 1;
            }
            int q = tid / LANES;
            int lane = tid - q * LANES;
            int per = dh / LANES;
            ctx.localBarrier();
            for (int k = 0; k < kEnd; k++) {
                if (tid < dh) {
                    int ko = (r * n + k) * 3 * d + hh * dh + tid;
                    kv[tid] = qkv.get(ko + d);
                    kv[MAX_HEAD_DIM + tid] = qkv.get(ko + 2 * d);
                }
                ctx.localBarrier();
                float partial = 0f;
                for (int e = lane * per; e < lane * per + per; e++) {
                    partial += qs[q * MAX_HEAD_DIM + e] * kv[e];
                }
                part[tid] = partial;
                ctx.localBarrier();
                if (lane == 0 && q < qn) {
                    float dot = 0f;
                    for (int l = 0; l < LANES; l++) {
                        dot += part[q * LANES + l];
                    }
                    int qi = q0 + q;
                    int valid = 0;
                    if (k < enc) {
                        valid = keyValid.get(r * enc + k);
                    }
                    // int masks, not short-circuit conditions (works around a TornadoVM codegen issue)
                    int allowed;
                    if (qi < enc) {
                        int self = k == qi ? 1 : 0;
                        int causal = k <= qi ? 1 : 0;
                        int before = k < enc ? 1 : 0;
                        allowed = before * TornadoMath.max(self, valid * TornadoMath.max(causal, isKnown));
                    } else {
                        int ctxKey = (k < ll ? 1 : 0) * valid;
                        int decKey = (k >= enc ? 1 : 0) * (k <= qi ? 1 : 0);
                        allowed = TornadoMath.max(ctxKey, decKey);
                    }
                    if (allowed != 0) {
                        float sc = dot * scale;
                        float mn = TornadoMath.max(mRun[q], sc);
                        float c = TornadoMath.exp(mRun[q] - mn);
                        float e = TornadoMath.exp(sc - mn);
                        lRun[q] = lRun[q] * c + e;
                        mRun[q] = mn;
                        corr[q] = c;
                        pw[q] = e;
                    } else {
                        corr[q] = 1f;
                        pw[q] = 0f;
                    }
                }
                ctx.localBarrier();
                if (tid < dh) {
                    float vj = kv[MAX_HEAD_DIM + tid];
                    a0 = a0 * corr[0] + pw[0] * vj;
                    a1 = a1 * corr[1] + pw[1] * vj;
                    a2 = a2 * corr[2] + pw[2] * vj;
                    a3 = a3 * corr[3] + pw[3] * vj;
                    a4 = a4 * corr[4] + pw[4] * vj;
                    a5 = a5 * corr[5] + pw[5] * vj;
                    a6 = a6 * corr[6] + pw[6] * vj;
                    a7 = a7 * corr[7] + pw[7] * vj;
                    a8 = a8 * corr[8] + pw[8] * vj;
                    a9 = a9 * corr[9] + pw[9] * vj;
                    a10 = a10 * corr[10] + pw[10] * vj;
                    a11 = a11 * corr[11] + pw[11] * vj;
                    a12 = a12 * corr[12] + pw[12] * vj;
                    a13 = a13 * corr[13] + pw[13] * vj;
                    a14 = a14 * corr[14] + pw[14] * vj;
                    a15 = a15 * corr[15] + pw[15] * vj;

                }
            }
            if (tid < dh) {
                if (0 < qn) {
                    out.set((r * n + q0 + 0) * d + hh * dh + tid, a0 / lRun[0]);
                }
                if (1 < qn) {
                    out.set((r * n + q0 + 1) * d + hh * dh + tid, a1 / lRun[1]);
                }
                if (2 < qn) {
                    out.set((r * n + q0 + 2) * d + hh * dh + tid, a2 / lRun[2]);
                }
                if (3 < qn) {
                    out.set((r * n + q0 + 3) * d + hh * dh + tid, a3 / lRun[3]);
                }
                if (4 < qn) {
                    out.set((r * n + q0 + 4) * d + hh * dh + tid, a4 / lRun[4]);
                }
                if (5 < qn) {
                    out.set((r * n + q0 + 5) * d + hh * dh + tid, a5 / lRun[5]);
                }
                if (6 < qn) {
                    out.set((r * n + q0 + 6) * d + hh * dh + tid, a6 / lRun[6]);
                }
                if (7 < qn) {
                    out.set((r * n + q0 + 7) * d + hh * dh + tid, a7 / lRun[7]);
                }
                if (8 < qn) {
                    out.set((r * n + q0 + 8) * d + hh * dh + tid, a8 / lRun[8]);
                }
                if (9 < qn) {
                    out.set((r * n + q0 + 9) * d + hh * dh + tid, a9 / lRun[9]);
                }
                if (10 < qn) {
                    out.set((r * n + q0 + 10) * d + hh * dh + tid, a10 / lRun[10]);
                }
                if (11 < qn) {
                    out.set((r * n + q0 + 11) * d + hh * dh + tid, a11 / lRun[11]);
                }
                if (12 < qn) {
                    out.set((r * n + q0 + 12) * d + hh * dh + tid, a12 / lRun[12]);
                }
                if (13 < qn) {
                    out.set((r * n + q0 + 13) * d + hh * dh + tid, a13 / lRun[13]);
                }
                if (14 < qn) {
                    out.set((r * n + q0 + 14) * d + hh * dh + tid, a14 / lRun[14]);
                }
                if (15 < qn) {
                    out.set((r * n + q0 + 15) * d + hh * dh + tid, a15 / lRun[15]);
                }
            }
        }
    }

    /**
     * {@link #timeAttention} on tensor cores with warp-level {@code mma} (CUDA only, compute capability
     * 8.0+), flash attention in the FlashAttention-2 layout: one work-group of {@link #MMA_THREADS}
     * threads per (row, head, {@link #MMA_QUERIES} consecutive query tokens), each warp 16 of the queries.
     * Each lane packs its Q fragments from {@code qkv} once; per block of {@link #MMA_KEYS} keys the
     * work-group stages K and V in shared memory as FP16 pairs (V transposed), each warp computes its
     * 16 × {@link #MMA_KEYS} scores with {@code mma}, masks them as {@link #timeAttention} does, folds
     * them into its online softmax in registers and feeds the probabilities straight back as the A
     * fragments of P·V. The output accumulates in FP32 registers. Heads must be {@link #MMA_HEAD_DIM}
     * wide. Writes {@code out} as {@link #timeAttention} does, and its FP16 twin {@code th} when
     * {@code twin} is 1.
     *
     * <p>The scores take Q and K as FP16 high and low parts ({@code Qh·Kh + Ql·Kh + Qh·Kl}), which keeps
     * them close to FP32: with Q and K rounded to FP16 alone, multi-block forecasts of the smallest
     * checkpoint drift past the reduced-precision bound.
     */
    public static void timeAttentionMma(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray keyValid, IntArray known,
            int rows, int enc, int ll, int n, int heads, int d, HalfFloatArray th, int twin) {
        int qTiles = (n + MMA_QUERIES - 1) / MMA_QUERIES;
        if (ctx.groupIdx < rows * heads * qTiles) {
            int[] ks = ctx.allocateIntLocalArray(MMA_KEYS * MMA_K_PITCH);
            int[] vs = ctx.allocateIntLocalArray(MMA_HEAD_DIM * MMA_V_PITCH);
            int[] ksl = ctx.allocateIntLocalArray(MMA_KEYS * MMA_K_PITCH);
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
            float scale = 0.125f;
            int qBase = r * n * 3 * d + hh * MMA_HEAD_DIM;
            int qa = qBase + ra * 3 * d;
            int qb = qBase + rb * 3 * d;
            boolean okA = ra < n;
            boolean okB = rb < n;
            int isKnown = known.get(r);
            // the keys any query of the work-group reads: an encoder-only tile stops at its last query, or the encoder
            int lastQ = TornadoMath.min(qt * MMA_QUERIES + MMA_QUERIES, n) - 1;
            int kEnd = n;
            if (lastQ < enc) {
                kEnd = isKnown == 1 ? enc : lastQ + 1;
            }
            float qx00 = okA ? qkv.get(qa + 0 + c) : 0f;
            float qy00 = okA ? qkv.get(qa + 0 + c + 1) : 0f;
            int q00 = pack(qx00, qy00);
            int ql00 = pack(lo(qx00), lo(qy00));
            float qx01 = okB ? qkv.get(qb + 0 + c) : 0f;
            float qy01 = okB ? qkv.get(qb + 0 + c + 1) : 0f;
            int q01 = pack(qx01, qy01);
            int ql01 = pack(lo(qx01), lo(qy01));
            float qx02 = okA ? qkv.get(qa + 8 + c) : 0f;
            float qy02 = okA ? qkv.get(qa + 8 + c + 1) : 0f;
            int q02 = pack(qx02, qy02);
            int ql02 = pack(lo(qx02), lo(qy02));
            float qx03 = okB ? qkv.get(qb + 8 + c) : 0f;
            float qy03 = okB ? qkv.get(qb + 8 + c + 1) : 0f;
            int q03 = pack(qx03, qy03);
            int ql03 = pack(lo(qx03), lo(qy03));
            float qx10 = okA ? qkv.get(qa + 16 + c) : 0f;
            float qy10 = okA ? qkv.get(qa + 16 + c + 1) : 0f;
            int q10 = pack(qx10, qy10);
            int ql10 = pack(lo(qx10), lo(qy10));
            float qx11 = okB ? qkv.get(qb + 16 + c) : 0f;
            float qy11 = okB ? qkv.get(qb + 16 + c + 1) : 0f;
            int q11 = pack(qx11, qy11);
            int ql11 = pack(lo(qx11), lo(qy11));
            float qx12 = okA ? qkv.get(qa + 24 + c) : 0f;
            float qy12 = okA ? qkv.get(qa + 24 + c + 1) : 0f;
            int q12 = pack(qx12, qy12);
            int ql12 = pack(lo(qx12), lo(qy12));
            float qx13 = okB ? qkv.get(qb + 24 + c) : 0f;
            float qy13 = okB ? qkv.get(qb + 24 + c + 1) : 0f;
            int q13 = pack(qx13, qy13);
            int ql13 = pack(lo(qx13), lo(qy13));
            float qx20 = okA ? qkv.get(qa + 32 + c) : 0f;
            float qy20 = okA ? qkv.get(qa + 32 + c + 1) : 0f;
            int q20 = pack(qx20, qy20);
            int ql20 = pack(lo(qx20), lo(qy20));
            float qx21 = okB ? qkv.get(qb + 32 + c) : 0f;
            float qy21 = okB ? qkv.get(qb + 32 + c + 1) : 0f;
            int q21 = pack(qx21, qy21);
            int ql21 = pack(lo(qx21), lo(qy21));
            float qx22 = okA ? qkv.get(qa + 40 + c) : 0f;
            float qy22 = okA ? qkv.get(qa + 40 + c + 1) : 0f;
            int q22 = pack(qx22, qy22);
            int ql22 = pack(lo(qx22), lo(qy22));
            float qx23 = okB ? qkv.get(qb + 40 + c) : 0f;
            float qy23 = okB ? qkv.get(qb + 40 + c + 1) : 0f;
            int q23 = pack(qx23, qy23);
            int ql23 = pack(lo(qx23), lo(qy23));
            float qx30 = okA ? qkv.get(qa + 48 + c) : 0f;
            float qy30 = okA ? qkv.get(qa + 48 + c + 1) : 0f;
            int q30 = pack(qx30, qy30);
            int ql30 = pack(lo(qx30), lo(qy30));
            float qx31 = okB ? qkv.get(qb + 48 + c) : 0f;
            float qy31 = okB ? qkv.get(qb + 48 + c + 1) : 0f;
            int q31 = pack(qx31, qy31);
            int ql31 = pack(lo(qx31), lo(qy31));
            float qx32 = okA ? qkv.get(qa + 56 + c) : 0f;
            float qy32 = okA ? qkv.get(qa + 56 + c + 1) : 0f;
            int q32 = pack(qx32, qy32);
            int ql32 = pack(lo(qx32), lo(qy32));
            float qx33 = okB ? qkv.get(qb + 56 + c) : 0f;
            float qy33 = okB ? qkv.get(qb + 56 + c + 1) : 0f;
            int q33 = pack(qx33, qy33);
            int ql33 = pack(lo(qx33), lo(qy33));
            float mA = MASKED;
            float mB = MASKED;
            float lA = 0f;
            float lB = 0f;
            float o00 = 0f, o01 = 0f, o02 = 0f, o03 = 0f;
            float o10 = 0f, o11 = 0f, o12 = 0f, o13 = 0f;
            float o20 = 0f, o21 = 0f, o22 = 0f, o23 = 0f;
            float o30 = 0f, o31 = 0f, o32 = 0f, o33 = 0f;
            float o40 = 0f, o41 = 0f, o42 = 0f, o43 = 0f;
            float o50 = 0f, o51 = 0f, o52 = 0f, o53 = 0f;
            float o60 = 0f, o61 = 0f, o62 = 0f, o63 = 0f;
            float o70 = 0f, o71 = 0f, o72 = 0f, o73 = 0f;
            for (int k0 = 0; k0 < kEnd; k0 += MMA_KEYS) {
                // K: key j, dimensions 2p, 2p + 1 at ks[j·pitch + p]; V transposed: dimension e, keys 2p, 2p + 1 at vs[e·pitch + p]
                for (int i = tid; i < MMA_KEYS * MMA_HEAD_DIM / 2; i += MMA_THREADS) {
                    int j = i >> 5;
                    int p = i & 31;
                    int ko = qBase + (k0 + j) * 3 * d + d + 2 * p;
                    boolean in = k0 + j < n;
                    float k0v = in ? qkv.get(ko) : 0f;
                    float k1v = in ? qkv.get(ko + 1) : 0f;
                    ks[j * MMA_K_PITCH + p] = pack(k0v, k1v);
                    ksl[j * MMA_K_PITCH + p] = pack(lo(k0v), lo(k1v));
                }
                for (int i = tid; i < MMA_KEYS * MMA_HEAD_DIM / 2; i += MMA_THREADS) {
                    int e = i & (MMA_HEAD_DIM - 1);
                    int p = i >> 6;
                    int vo = qBase + (k0 + 2 * p) * 3 * d + 2 * d + e;
                    float v0 = k0 + 2 * p < n ? qkv.get(vo) : 0f;
                    float v1 = k0 + 2 * p + 1 < n ? qkv.get(vo + 3 * d) : 0f;
                    vs[e * MMA_V_PITCH + p] = pack(v0, v1);
                }
                ctx.localBarrier();
                HalfFloat[] fq0 = ctx.mmaFragmentA(q00, q01, q02, q03);
                HalfFloat[] fql0 = ctx.mmaFragmentA(ql00, ql01, ql02, ql03);
                HalfFloat[] fq1 = ctx.mmaFragmentA(q10, q11, q12, q13);
                HalfFloat[] fql1 = ctx.mmaFragmentA(ql10, ql11, ql12, ql13);
                HalfFloat[] fq2 = ctx.mmaFragmentA(q20, q21, q22, q23);
                HalfFloat[] fql2 = ctx.mmaFragmentA(ql20, ql21, ql22, ql23);
                HalfFloat[] fq3 = ctx.mmaFragmentA(q30, q31, q32, q33);
                HalfFloat[] fql3 = ctx.mmaFragmentA(ql30, ql31, ql32, ql33);
                float[] s0 = ctx.mmaFragment(0f);
                int kr0 = (0 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk00 = ctx.mmaFragmentB(ks[kr0 + 0], ks[kr0 + 4]);
                s0 = ctx.mma(fq0, fk00, s0, MMAShape.M16N8K16);
                s0 = ctx.mma(fql0, fk00, s0, MMAShape.M16N8K16);
                s0 = ctx.mma(fq0, ctx.mmaFragmentB(ksl[kr0 + 0], ksl[kr0 + 4]), s0, MMAShape.M16N8K16);
                HalfFloat[] fk01 = ctx.mmaFragmentB(ks[kr0 + 8], ks[kr0 + 12]);
                s0 = ctx.mma(fq1, fk01, s0, MMAShape.M16N8K16);
                s0 = ctx.mma(fql1, fk01, s0, MMAShape.M16N8K16);
                s0 = ctx.mma(fq1, ctx.mmaFragmentB(ksl[kr0 + 8], ksl[kr0 + 12]), s0, MMAShape.M16N8K16);
                HalfFloat[] fk02 = ctx.mmaFragmentB(ks[kr0 + 16], ks[kr0 + 20]);
                s0 = ctx.mma(fq2, fk02, s0, MMAShape.M16N8K16);
                s0 = ctx.mma(fql2, fk02, s0, MMAShape.M16N8K16);
                s0 = ctx.mma(fq2, ctx.mmaFragmentB(ksl[kr0 + 16], ksl[kr0 + 20]), s0, MMAShape.M16N8K16);
                HalfFloat[] fk03 = ctx.mmaFragmentB(ks[kr0 + 24], ks[kr0 + 28]);
                s0 = ctx.mma(fq3, fk03, s0, MMAShape.M16N8K16);
                s0 = ctx.mma(fql3, fk03, s0, MMAShape.M16N8K16);
                s0 = ctx.mma(fq3, ctx.mmaFragmentB(ksl[kr0 + 24], ksl[kr0 + 28]), s0, MMAShape.M16N8K16);
                float[] s1 = ctx.mmaFragment(0f);
                int kr1 = (8 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk10 = ctx.mmaFragmentB(ks[kr1 + 0], ks[kr1 + 4]);
                s1 = ctx.mma(fq0, fk10, s1, MMAShape.M16N8K16);
                s1 = ctx.mma(fql0, fk10, s1, MMAShape.M16N8K16);
                s1 = ctx.mma(fq0, ctx.mmaFragmentB(ksl[kr1 + 0], ksl[kr1 + 4]), s1, MMAShape.M16N8K16);
                HalfFloat[] fk11 = ctx.mmaFragmentB(ks[kr1 + 8], ks[kr1 + 12]);
                s1 = ctx.mma(fq1, fk11, s1, MMAShape.M16N8K16);
                s1 = ctx.mma(fql1, fk11, s1, MMAShape.M16N8K16);
                s1 = ctx.mma(fq1, ctx.mmaFragmentB(ksl[kr1 + 8], ksl[kr1 + 12]), s1, MMAShape.M16N8K16);
                HalfFloat[] fk12 = ctx.mmaFragmentB(ks[kr1 + 16], ks[kr1 + 20]);
                s1 = ctx.mma(fq2, fk12, s1, MMAShape.M16N8K16);
                s1 = ctx.mma(fql2, fk12, s1, MMAShape.M16N8K16);
                s1 = ctx.mma(fq2, ctx.mmaFragmentB(ksl[kr1 + 16], ksl[kr1 + 20]), s1, MMAShape.M16N8K16);
                HalfFloat[] fk13 = ctx.mmaFragmentB(ks[kr1 + 24], ks[kr1 + 28]);
                s1 = ctx.mma(fq3, fk13, s1, MMAShape.M16N8K16);
                s1 = ctx.mma(fql3, fk13, s1, MMAShape.M16N8K16);
                s1 = ctx.mma(fq3, ctx.mmaFragmentB(ksl[kr1 + 24], ksl[kr1 + 28]), s1, MMAShape.M16N8K16);
                float[] s2 = ctx.mmaFragment(0f);
                int kr2 = (16 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk20 = ctx.mmaFragmentB(ks[kr2 + 0], ks[kr2 + 4]);
                s2 = ctx.mma(fq0, fk20, s2, MMAShape.M16N8K16);
                s2 = ctx.mma(fql0, fk20, s2, MMAShape.M16N8K16);
                s2 = ctx.mma(fq0, ctx.mmaFragmentB(ksl[kr2 + 0], ksl[kr2 + 4]), s2, MMAShape.M16N8K16);
                HalfFloat[] fk21 = ctx.mmaFragmentB(ks[kr2 + 8], ks[kr2 + 12]);
                s2 = ctx.mma(fq1, fk21, s2, MMAShape.M16N8K16);
                s2 = ctx.mma(fql1, fk21, s2, MMAShape.M16N8K16);
                s2 = ctx.mma(fq1, ctx.mmaFragmentB(ksl[kr2 + 8], ksl[kr2 + 12]), s2, MMAShape.M16N8K16);
                HalfFloat[] fk22 = ctx.mmaFragmentB(ks[kr2 + 16], ks[kr2 + 20]);
                s2 = ctx.mma(fq2, fk22, s2, MMAShape.M16N8K16);
                s2 = ctx.mma(fql2, fk22, s2, MMAShape.M16N8K16);
                s2 = ctx.mma(fq2, ctx.mmaFragmentB(ksl[kr2 + 16], ksl[kr2 + 20]), s2, MMAShape.M16N8K16);
                HalfFloat[] fk23 = ctx.mmaFragmentB(ks[kr2 + 24], ks[kr2 + 28]);
                s2 = ctx.mma(fq3, fk23, s2, MMAShape.M16N8K16);
                s2 = ctx.mma(fql3, fk23, s2, MMAShape.M16N8K16);
                s2 = ctx.mma(fq3, ctx.mmaFragmentB(ksl[kr2 + 24], ksl[kr2 + 28]), s2, MMAShape.M16N8K16);
                float[] s3 = ctx.mmaFragment(0f);
                int kr3 = (24 + (lane >> 2)) * MMA_K_PITCH + quad;
                HalfFloat[] fk30 = ctx.mmaFragmentB(ks[kr3 + 0], ks[kr3 + 4]);
                s3 = ctx.mma(fq0, fk30, s3, MMAShape.M16N8K16);
                s3 = ctx.mma(fql0, fk30, s3, MMAShape.M16N8K16);
                s3 = ctx.mma(fq0, ctx.mmaFragmentB(ksl[kr3 + 0], ksl[kr3 + 4]), s3, MMAShape.M16N8K16);
                HalfFloat[] fk31 = ctx.mmaFragmentB(ks[kr3 + 8], ks[kr3 + 12]);
                s3 = ctx.mma(fq1, fk31, s3, MMAShape.M16N8K16);
                s3 = ctx.mma(fql1, fk31, s3, MMAShape.M16N8K16);
                s3 = ctx.mma(fq1, ctx.mmaFragmentB(ksl[kr3 + 8], ksl[kr3 + 12]), s3, MMAShape.M16N8K16);
                HalfFloat[] fk32 = ctx.mmaFragmentB(ks[kr3 + 16], ks[kr3 + 20]);
                s3 = ctx.mma(fq2, fk32, s3, MMAShape.M16N8K16);
                s3 = ctx.mma(fql2, fk32, s3, MMAShape.M16N8K16);
                s3 = ctx.mma(fq2, ctx.mmaFragmentB(ksl[kr3 + 16], ksl[kr3 + 20]), s3, MMAShape.M16N8K16);
                HalfFloat[] fk33 = ctx.mmaFragmentB(ks[kr3 + 24], ks[kr3 + 28]);
                s3 = ctx.mma(fq3, fk33, s3, MMAShape.M16N8K16);
                s3 = ctx.mma(fql3, fk33, s3, MMAShape.M16N8K16);
                s3 = ctx.mma(fq3, ctx.mmaFragmentB(ksl[kr3 + 24], ksl[kr3 + 28]), s3, MMAShape.M16N8K16);
                int kv00 = keyOk(keyValid, r, enc, k0 + 0 + c);
                float x00 = allowed(ra, k0 + 0 + c, enc, ll, kv00, isKnown) != 0 ? s0[0] * scale : MASKED;
                float y00 = allowed(rb, k0 + 0 + c, enc, ll, kv00, isKnown) != 0 ? s0[2] * scale : MASKED;
                int kv01 = keyOk(keyValid, r, enc, k0 + 1 + c);
                float x01 = allowed(ra, k0 + 1 + c, enc, ll, kv01, isKnown) != 0 ? s0[1] * scale : MASKED;
                float y01 = allowed(rb, k0 + 1 + c, enc, ll, kv01, isKnown) != 0 ? s0[3] * scale : MASKED;
                int kv10 = keyOk(keyValid, r, enc, k0 + 8 + c);
                float x10 = allowed(ra, k0 + 8 + c, enc, ll, kv10, isKnown) != 0 ? s1[0] * scale : MASKED;
                float y10 = allowed(rb, k0 + 8 + c, enc, ll, kv10, isKnown) != 0 ? s1[2] * scale : MASKED;
                int kv11 = keyOk(keyValid, r, enc, k0 + 9 + c);
                float x11 = allowed(ra, k0 + 9 + c, enc, ll, kv11, isKnown) != 0 ? s1[1] * scale : MASKED;
                float y11 = allowed(rb, k0 + 9 + c, enc, ll, kv11, isKnown) != 0 ? s1[3] * scale : MASKED;
                int kv20 = keyOk(keyValid, r, enc, k0 + 16 + c);
                float x20 = allowed(ra, k0 + 16 + c, enc, ll, kv20, isKnown) != 0 ? s2[0] * scale : MASKED;
                float y20 = allowed(rb, k0 + 16 + c, enc, ll, kv20, isKnown) != 0 ? s2[2] * scale : MASKED;
                int kv21 = keyOk(keyValid, r, enc, k0 + 17 + c);
                float x21 = allowed(ra, k0 + 17 + c, enc, ll, kv21, isKnown) != 0 ? s2[1] * scale : MASKED;
                float y21 = allowed(rb, k0 + 17 + c, enc, ll, kv21, isKnown) != 0 ? s2[3] * scale : MASKED;
                int kv30 = keyOk(keyValid, r, enc, k0 + 24 + c);
                float x30 = allowed(ra, k0 + 24 + c, enc, ll, kv30, isKnown) != 0 ? s3[0] * scale : MASKED;
                float y30 = allowed(rb, k0 + 24 + c, enc, ll, kv30, isKnown) != 0 ? s3[2] * scale : MASKED;
                int kv31 = keyOk(keyValid, r, enc, k0 + 25 + c);
                float x31 = allowed(ra, k0 + 25 + c, enc, ll, kv31, isKnown) != 0 ? s3[1] * scale : MASKED;
                float y31 = allowed(rb, k0 + 25 + c, enc, ll, kv31, isKnown) != 0 ? s3[3] * scale : MASKED;
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
                HalfFloat[] fv00 = ctx.mmaFragmentB(vs[vr0 + 0], vs[vr0 + 4]);
                t0 = ctx.mma(fp0, fv00, t0, MMAShape.M16N8K16);
                HalfFloat[] fv01 = ctx.mmaFragmentB(vs[vr0 + 8], vs[vr0 + 12]);
                t0 = ctx.mma(fp1, fv01, t0, MMAShape.M16N8K16);
                o00 = o00 * cA + t0[0];
                o01 = o01 * cA + t0[1];
                o02 = o02 * cB + t0[2];
                o03 = o03 * cB + t0[3];
                int vr1 = (8 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t1 = ctx.mmaFragment(0f);
                HalfFloat[] fv10 = ctx.mmaFragmentB(vs[vr1 + 0], vs[vr1 + 4]);
                t1 = ctx.mma(fp0, fv10, t1, MMAShape.M16N8K16);
                HalfFloat[] fv11 = ctx.mmaFragmentB(vs[vr1 + 8], vs[vr1 + 12]);
                t1 = ctx.mma(fp1, fv11, t1, MMAShape.M16N8K16);
                o10 = o10 * cA + t1[0];
                o11 = o11 * cA + t1[1];
                o12 = o12 * cB + t1[2];
                o13 = o13 * cB + t1[3];
                int vr2 = (16 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t2 = ctx.mmaFragment(0f);
                HalfFloat[] fv20 = ctx.mmaFragmentB(vs[vr2 + 0], vs[vr2 + 4]);
                t2 = ctx.mma(fp0, fv20, t2, MMAShape.M16N8K16);
                HalfFloat[] fv21 = ctx.mmaFragmentB(vs[vr2 + 8], vs[vr2 + 12]);
                t2 = ctx.mma(fp1, fv21, t2, MMAShape.M16N8K16);
                o20 = o20 * cA + t2[0];
                o21 = o21 * cA + t2[1];
                o22 = o22 * cB + t2[2];
                o23 = o23 * cB + t2[3];
                int vr3 = (24 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t3 = ctx.mmaFragment(0f);
                HalfFloat[] fv30 = ctx.mmaFragmentB(vs[vr3 + 0], vs[vr3 + 4]);
                t3 = ctx.mma(fp0, fv30, t3, MMAShape.M16N8K16);
                HalfFloat[] fv31 = ctx.mmaFragmentB(vs[vr3 + 8], vs[vr3 + 12]);
                t3 = ctx.mma(fp1, fv31, t3, MMAShape.M16N8K16);
                o30 = o30 * cA + t3[0];
                o31 = o31 * cA + t3[1];
                o32 = o32 * cB + t3[2];
                o33 = o33 * cB + t3[3];
                int vr4 = (32 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t4 = ctx.mmaFragment(0f);
                HalfFloat[] fv40 = ctx.mmaFragmentB(vs[vr4 + 0], vs[vr4 + 4]);
                t4 = ctx.mma(fp0, fv40, t4, MMAShape.M16N8K16);
                HalfFloat[] fv41 = ctx.mmaFragmentB(vs[vr4 + 8], vs[vr4 + 12]);
                t4 = ctx.mma(fp1, fv41, t4, MMAShape.M16N8K16);
                o40 = o40 * cA + t4[0];
                o41 = o41 * cA + t4[1];
                o42 = o42 * cB + t4[2];
                o43 = o43 * cB + t4[3];
                int vr5 = (40 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t5 = ctx.mmaFragment(0f);
                HalfFloat[] fv50 = ctx.mmaFragmentB(vs[vr5 + 0], vs[vr5 + 4]);
                t5 = ctx.mma(fp0, fv50, t5, MMAShape.M16N8K16);
                HalfFloat[] fv51 = ctx.mmaFragmentB(vs[vr5 + 8], vs[vr5 + 12]);
                t5 = ctx.mma(fp1, fv51, t5, MMAShape.M16N8K16);
                o50 = o50 * cA + t5[0];
                o51 = o51 * cA + t5[1];
                o52 = o52 * cB + t5[2];
                o53 = o53 * cB + t5[3];
                int vr6 = (48 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t6 = ctx.mmaFragment(0f);
                HalfFloat[] fv60 = ctx.mmaFragmentB(vs[vr6 + 0], vs[vr6 + 4]);
                t6 = ctx.mma(fp0, fv60, t6, MMAShape.M16N8K16);
                HalfFloat[] fv61 = ctx.mmaFragmentB(vs[vr6 + 8], vs[vr6 + 12]);
                t6 = ctx.mma(fp1, fv61, t6, MMAShape.M16N8K16);
                o60 = o60 * cA + t6[0];
                o61 = o61 * cA + t6[1];
                o62 = o62 * cB + t6[2];
                o63 = o63 * cB + t6[3];
                int vr7 = (56 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t7 = ctx.mmaFragment(0f);
                HalfFloat[] fv70 = ctx.mmaFragmentB(vs[vr7 + 0], vs[vr7 + 4]);
                t7 = ctx.mma(fp0, fv70, t7, MMAShape.M16N8K16);
                HalfFloat[] fv71 = ctx.mmaFragmentB(vs[vr7 + 8], vs[vr7 + 12]);
                t7 = ctx.mma(fp1, fv71, t7, MMAShape.M16N8K16);
                o70 = o70 * cA + t7[0];
                o71 = o71 * cA + t7[1];
                o72 = o72 * cB + t7[2];
                o73 = o73 * cB + t7[3];
                ctx.localBarrier();
            }
            lA = lA + ctx.simdShuffle(lA, lane ^ 1);
            lA = lA + ctx.simdShuffle(lA, lane ^ 2);
            lB = lB + ctx.simdShuffle(lB, lane ^ 1);
            lB = lB + ctx.simdShuffle(lB, lane ^ 2);
            // every query reads itself, so l > 0 for every real query
            float iA = 1f / lA;
            float iB = 1f / lB;
            int oa = (r * n + ra) * d + hh * MMA_HEAD_DIM + c;
            int ob = (r * n + rb) * d + hh * MMA_HEAD_DIM + c;
            if (okA) {
                store(out, th, twin, oa + 0, o00 * iA);
                store(out, th, twin, oa + 1, o01 * iA);
                store(out, th, twin, oa + 8, o10 * iA);
                store(out, th, twin, oa + 9, o11 * iA);
                store(out, th, twin, oa + 16, o20 * iA);
                store(out, th, twin, oa + 17, o21 * iA);
                store(out, th, twin, oa + 24, o30 * iA);
                store(out, th, twin, oa + 25, o31 * iA);
                store(out, th, twin, oa + 32, o40 * iA);
                store(out, th, twin, oa + 33, o41 * iA);
                store(out, th, twin, oa + 40, o50 * iA);
                store(out, th, twin, oa + 41, o51 * iA);
                store(out, th, twin, oa + 48, o60 * iA);
                store(out, th, twin, oa + 49, o61 * iA);
                store(out, th, twin, oa + 56, o70 * iA);
                store(out, th, twin, oa + 57, o71 * iA);
            }
            if (okB) {
                store(out, th, twin, ob + 0, o02 * iB);
                store(out, th, twin, ob + 1, o03 * iB);
                store(out, th, twin, ob + 8, o12 * iB);
                store(out, th, twin, ob + 9, o13 * iB);
                store(out, th, twin, ob + 16, o22 * iB);
                store(out, th, twin, ob + 17, o23 * iB);
                store(out, th, twin, ob + 24, o32 * iB);
                store(out, th, twin, ob + 25, o33 * iB);
                store(out, th, twin, ob + 32, o42 * iB);
                store(out, th, twin, ob + 33, o43 * iB);
                store(out, th, twin, ob + 40, o52 * iB);
                store(out, th, twin, ob + 41, o53 * iB);
                store(out, th, twin, ob + 48, o62 * iB);
                store(out, th, twin, ob + 49, o63 * iB);
                store(out, th, twin, ob + 56, o72 * iB);
                store(out, th, twin, ob + 57, o73 * iB);
            }
        }
    }

    /** Whether a query at {@code qi} reads the key at {@code k} ({@code BlockLayout.timeAllowed}), as 0 or 1. */
    static int allowed(int qi, int k, int enc, int ll, int valid, int isKnown) {
        // int masks, not short-circuit conditions (works around a TornadoVM codegen issue)
        int allowed;
        if (qi < enc) {
            int self = k == qi ? 1 : 0;
            int causal = k <= qi ? 1 : 0;
            int before = k < enc ? 1 : 0;
            allowed = before * TornadoMath.max(self, valid * TornadoMath.max(causal, isKnown));
        } else {
            int ctxKey = (k < ll ? 1 : 0) * valid;
            int decKey = (k >= enc ? 1 : 0) * (k <= qi ? 1 : 0);
            allowed = TornadoMath.max(ctxKey, decKey);
        }
        return allowed;
    }

    /** {@code keyValid [r·enc + k]} for an encoder key, 0 for any other. */
    static int keyOk(IntArray keyValid, int r, int enc, int k) {
        int v = 0;
        if (k < enc) {
            v = keyValid.get(r * enc + k);
        }
        return v;
    }

    /** {@code out[i] = v}, and its FP16 twin, saturated to ±{@code HALF_MAX}, when {@code twin} is 1. */
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

    /** What FP16 rounding loses of {@code x}: {@code x = half(x) + lo(x)} to about 2⁻²². */
    static float lo(float x) {
        return x - new HalfFloat(x).getFloat32();
    }

    /**
     * Attention across the rows of each series, per token slot and head (flash style): one
     * work-group per ({@link #ATTN_QUERIES} consecutive rows, slot, head), with keys over the union of
     * the rows' series ({@code groupStart .. groupEnd}, which must be contiguous runs of rows).
     *
     * <p>Masks ({@code BlockLayout.variateAllowed}): a row reads the rows of its series, at an encoder
     * slot only where the key row's patch is a valid key; with {@code covariates} a covariate row
     * never reads a target row ({@code role 0}) and a known-covariate row ({@code role 2}) never reads
     * another kind. A row always reads itself.
     */
    public static void variateAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray keyValid,
            IntArray groupStart, IntArray groupEnd, IntArray role, int covariates, int rows, int enc, int n, int heads, int dh,
            int d) {
        if (ctx.groupIdx < n * heads * ((rows + ATTN_QUERIES - 1) / ATTN_QUERIES)) {
            int wg = ctx.groupIdx;
            int tid = ctx.localIdx;
            int tiles = (rows + ATTN_QUERIES - 1) / ATTN_QUERIES;
            int tile = wg % tiles;
            int hh = (wg / tiles) % heads;
            int t = wg / (tiles * heads);
            int q0 = tile * ATTN_QUERIES;
            int qn = TornadoMath.min(ATTN_QUERIES, rows - q0);
            float scale = 1f / TornadoMath.sqrt(dh);
            float[] qs = ctx.allocateFloatLocalArray(ATTN_QUERIES * MAX_HEAD_DIM);
            float[] kv = ctx.allocateFloatLocalArray(2 * MAX_HEAD_DIM);
            float[] part = ctx.allocateFloatLocalArray(ATTN_THREADS);
            float[] mRun = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] lRun = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] corr = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] pw = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            // one scalar per query of ATTN_QUERIES = 16 (not an array), so every kernel compiler keeps them in registers
            float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f, a4 = 0f, a5 = 0f, a6 = 0f, a7 = 0f; float a8 = 0f, a9 = 0f, a10 = 0f, a11 = 0f, a12 = 0f, a13 = 0f, a14 = 0f, a15 = 0f;
            for (int qq = 0; qq < ATTN_QUERIES; qq++) {
                if (qq < qn) {
                    for (int e = tid; e < dh; e += ATTN_THREADS) {
                        qs[qq * MAX_HEAD_DIM + e] = qkv.get(((q0 + qq) * n + t) * 3 * d + hh * dh + e);
                    }
                }
            }
            if (tid < ATTN_QUERIES) {
                mRun[tid] = -3.4e38f;
                lRun[tid] = 0f;
            }
            int kStart = groupStart.get(q0);
            int kEnd = groupEnd.get(q0 + qn - 1);
            int q = tid / LANES;
            int lane = tid - q * LANES;
            int per = dh / LANES;
            ctx.localBarrier();
            for (int k = kStart; k < kEnd; k++) {
                if (tid < dh) {
                    int ko = (k * n + t) * 3 * d + hh * dh + tid;
                    kv[tid] = qkv.get(ko + d);
                    kv[MAX_HEAD_DIM + tid] = qkv.get(ko + 2 * d);
                }
                ctx.localBarrier();
                float partial = 0f;
                for (int e = lane * per; e < lane * per + per; e++) {
                    partial += qs[q * MAX_HEAD_DIM + e] * kv[e];
                }
                part[tid] = partial;
                ctx.localBarrier();
                if (lane == 0 && q < qn) {
                    float dot = 0f;
                    for (int l = 0; l < LANES; l++) {
                        dot += part[q * LANES + l];
                    }
                    int qi = q0 + q;
                    int inGroup = (k >= groupStart.get(qi) ? 1 : 0) * (k < groupEnd.get(qi) ? 1 : 0);
                    int self = k == qi ? 1 : 0;
                    int valid = 1;
                    if (t < enc) {
                        valid = keyValid.get(k * enc + t);
                    }
                    int rq = role.get(qi);
                    int rk = role.get(k);
                    int forbid = covariates * TornadoMath.max((rq != 0 ? 1 : 0) * (rk == 0 ? 1 : 0),
                            (rq == 2 ? 1 : 0) * (rk != 2 ? 1 : 0));
                    int allowed = inGroup * TornadoMath.max(self, valid * (1 - forbid));
                    if (allowed != 0) {
                        float sc = dot * scale;
                        float mn = TornadoMath.max(mRun[q], sc);
                        float c = TornadoMath.exp(mRun[q] - mn);
                        float e = TornadoMath.exp(sc - mn);
                        lRun[q] = lRun[q] * c + e;
                        mRun[q] = mn;
                        corr[q] = c;
                        pw[q] = e;
                    } else {
                        corr[q] = 1f;
                        pw[q] = 0f;
                    }
                }
                ctx.localBarrier();
                if (tid < dh) {
                    float vj = kv[MAX_HEAD_DIM + tid];
                    a0 = a0 * corr[0] + pw[0] * vj;
                    a1 = a1 * corr[1] + pw[1] * vj;
                    a2 = a2 * corr[2] + pw[2] * vj;
                    a3 = a3 * corr[3] + pw[3] * vj;
                    a4 = a4 * corr[4] + pw[4] * vj;
                    a5 = a5 * corr[5] + pw[5] * vj;
                    a6 = a6 * corr[6] + pw[6] * vj;
                    a7 = a7 * corr[7] + pw[7] * vj;
                    a8 = a8 * corr[8] + pw[8] * vj;
                    a9 = a9 * corr[9] + pw[9] * vj;
                    a10 = a10 * corr[10] + pw[10] * vj;
                    a11 = a11 * corr[11] + pw[11] * vj;
                    a12 = a12 * corr[12] + pw[12] * vj;
                    a13 = a13 * corr[13] + pw[13] * vj;
                    a14 = a14 * corr[14] + pw[14] * vj;
                    a15 = a15 * corr[15] + pw[15] * vj;

                }
            }
            if (tid < dh) {
                if (0 < qn) {
                    out.set(((q0 + 0) * n + t) * d + hh * dh + tid, a0 / lRun[0]);
                }
                if (1 < qn) {
                    out.set(((q0 + 1) * n + t) * d + hh * dh + tid, a1 / lRun[1]);
                }
                if (2 < qn) {
                    out.set(((q0 + 2) * n + t) * d + hh * dh + tid, a2 / lRun[2]);
                }
                if (3 < qn) {
                    out.set(((q0 + 3) * n + t) * d + hh * dh + tid, a3 / lRun[3]);
                }
                if (4 < qn) {
                    out.set(((q0 + 4) * n + t) * d + hh * dh + tid, a4 / lRun[4]);
                }
                if (5 < qn) {
                    out.set(((q0 + 5) * n + t) * d + hh * dh + tid, a5 / lRun[5]);
                }
                if (6 < qn) {
                    out.set(((q0 + 6) * n + t) * d + hh * dh + tid, a6 / lRun[6]);
                }
                if (7 < qn) {
                    out.set(((q0 + 7) * n + t) * d + hh * dh + tid, a7 / lRun[7]);
                }
                if (8 < qn) {
                    out.set(((q0 + 8) * n + t) * d + hh * dh + tid, a8 / lRun[8]);
                }
                if (9 < qn) {
                    out.set(((q0 + 9) * n + t) * d + hh * dh + tid, a9 / lRun[9]);
                }
                if (10 < qn) {
                    out.set(((q0 + 10) * n + t) * d + hh * dh + tid, a10 / lRun[10]);
                }
                if (11 < qn) {
                    out.set(((q0 + 11) * n + t) * d + hh * dh + tid, a11 / lRun[11]);
                }
                if (12 < qn) {
                    out.set(((q0 + 12) * n + t) * d + hh * dh + tid, a12 / lRun[12]);
                }
                if (13 < qn) {
                    out.set(((q0 + 13) * n + t) * d + hh * dh + tid, a13 / lRun[13]);
                }
                if (14 < qn) {
                    out.set(((q0 + 14) * n + t) * d + hh * dh + tid, a14 / lRun[14]);
                }
                if (15 < qn) {
                    out.set(((q0 + 15) * n + t) * d + hh * dh + tid, a15 / lRun[15]);
                }
            }
        }
    }

    /** Threads per work-group of {@link #attentionSoftmax}. */
    public static final int SOFTMAX_THREADS = 128;

    /**
     * Copies the q, k and v of the batches {@code b0 .. b0 + count} (batch {@code b = row·heads + head})
     * from the token-major {@code qkv} into heads-major {@code [b − b0][token][dh]} arrays, for time
     * attention as batched GEMMs; batches past {@code rows·heads} are zero. One thread per element.
     */
    public static void splitHeads(KernelContext ctx, FloatArray qkv, FloatArray qh, FloatArray kh, FloatArray vh, int b0, int count,
            int rows, int n, int heads, int dh, int d) {
        int id = ctx.globalIdx;
        if (id < count * n * dh) {
            int e = id % dh;
            int t = (id / dh) % n;
            int b = b0 + id / (n * dh);
            float q = 0f;
            float k = 0f;
            float v = 0f;
            if (b < rows * heads) {
                int base = ((b / heads) * n + t) * 3 * d + (b % heads) * dh + e;
                q = qkv.get(base);
                k = qkv.get(base + d);
                v = qkv.get(base + 2 * d);
            }
            qh.set(id, q);
            kh.set(id, k);
            vh.set(id, v);
        }
    }

    /**
     * Masked softmax of time-attention scores in place: {@code p [(b − b0)·n + query]·n + key]}, scaled
     * scores in, probabilities out, 0 where {@code BlockLayout.timeAllowed} forbids the key (the masks
     * of {@link #timeAttention}). One work-group of {@link #SOFTMAX_THREADS} per (batch, query).
     */
    public static void attentionSoftmax(KernelContext ctx, FloatArray p, IntArray keyValid, IntArray known, int b0, int count,
            int rows, int enc, int ll, int n, int heads) {
        if (ctx.groupIdx < count * n) {
            int wg = ctx.groupIdx;
            int tid = ctx.localIdx;
            float[] red = ctx.allocateFloatLocalArray(SOFTMAX_THREADS);
            int lb = wg / n;
            int qi = wg - lb * n;
            int b = b0 + lb;
            int r = TornadoMath.min(b / heads, rows - 1);
            int isKnown = known.get(r);
            int base = (lb * n + qi) * n;
            float mx = -3.4e38f;
            for (int k = tid; k < n; k += SOFTMAX_THREADS) {
                int valid = 0;
                if (k < enc) {
                    valid = keyValid.get(r * enc + k);
                }
                // int masks, not short-circuit conditions (works around a TornadoVM codegen issue)
                int allowed;
                if (qi < enc) {
                    int self = k == qi ? 1 : 0;
                    int causal = k <= qi ? 1 : 0;
                    int before = k < enc ? 1 : 0;
                    allowed = before * TornadoMath.max(self, valid * TornadoMath.max(causal, isKnown));
                } else {
                    int ctxKey = (k < ll ? 1 : 0) * valid;
                    int decKey = (k >= enc ? 1 : 0) * (k <= qi ? 1 : 0);
                    allowed = TornadoMath.max(ctxKey, decKey);
                }
                float sc = p.get(base + k);
                if (allowed != 0) {
                    mx = TornadoMath.max(mx, sc);
                } else {
                    p.set(base + k, -3.4e38f);
                }
            }
            red[tid] = mx;
            ctx.localBarrier();
            for (int s = SOFTMAX_THREADS / 2; s > 0; s /= 2) {
                if (tid < s) {
                    red[tid] = TornadoMath.max(red[tid], red[tid + s]);
                }
                ctx.localBarrier();
            }
            float m = red[0];
            ctx.localBarrier();
            float sum = 0f;
            for (int k = tid; k < n; k += SOFTMAX_THREADS) {
                float sc = p.get(base + k);
                float e = 0f;
                if (sc > -3.0e38f) {
                    e = TornadoMath.exp(sc - m);
                }
                p.set(base + k, e);
                sum += e;
            }
            red[tid] = sum;
            ctx.localBarrier();
            for (int s = SOFTMAX_THREADS / 2; s > 0; s /= 2) {
                if (tid < s) {
                    red[tid] = red[tid] + red[tid + s];
                }
                ctx.localBarrier();
            }
            float inv = 1f / red[0];
            for (int k = tid; k < n; k += SOFTMAX_THREADS) {
                p.set(base + k, p.get(base + k) * inv);
            }
        }
    }

    /**
     * Writes the heads-major attention outputs {@code oh [(b − b0)·n + token]·dh + j]} of the batches
     * {@code b0 .. b0 + count} back to {@code out [(row·n + token)·d + head·dh + j]}. One thread per element.
     */
    public static void mergeHeads(KernelContext ctx, FloatArray oh, FloatArray out, int b0, int count, int rows, int n, int heads,
            int dh, int d) {
        int id = ctx.globalIdx;
        if (id < count * n * dh) {
            int e = id % dh;
            int t = (id / dh) % n;
            int b = b0 + id / (n * dh);
            if (b < rows * heads) {
                out.set(((b / heads) * n + t) * d + (b % heads) * dh + e, oh.get(id));
            }
        }
    }

    /**
     * {@code att[i] ·= sigmoid(g[i])} for the first {@code count} elements (the attention output gate), and
     * its FP16 twin {@code th} when {@code twin} is 1.
     */
    public static void gate(KernelContext ctx, FloatArray att, FloatArray g, int count, HalfFloatArray th, int twin) {
        int i = ctx.globalIdx;
        if (i < count) {
            store(att, th, twin, i, att.get(i) / (1f + TornadoMath.exp(-g.get(i))));
        }
    }

    /** {@code dec [(r·h + i)·d + j] = y [(r·n + enc + i)·d + j]}: the decoder tokens, for the head. */
    public static void gatherDecoder(KernelContext ctx, FloatArray y, FloatArray dec, int rows, int enc, int n, int h, int d) {
        int id = ctx.globalIdx;
        if (id < rows * h * d) {
            int j = id % d;
            int tok = id / d;
            int i = tok % h;
            int r = tok / h;
            dec.set(id, y.get((r * n + enc + i) * d + j));
        }
    }
}
