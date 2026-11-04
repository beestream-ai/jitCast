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
package ai.beestream.jitcast.chronos2.tornado;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The TornadoVM kernels specific to Chronos-2 (portable: OpenCL, Metal, CUDA): token assembly,
 * RoPE, time and group attention with the {@code ModelLayout} masks, and the future-token gather.
 * The GEMMs (with their ReLU epilogue) and RMSNorm are the shared {@code tornado.Kernels}. Each
 * kernel is a device version of a {@code chronos2.cpu.CpuChronos2} stage and is tested against it.
 *
 * <p>Activations are row-major {@code [rows · tokens · width]}, each row's context tokens, then
 * {@code [REG]}, then its future tokens; {@code q|k|v} is heads-major per token. The attention
 * kernels follow the Timestral ones (flash style, one work-group per block of queries), without
 * the {@code 1/√d_kv} scaling, as the reference; the time attention takes blocks of keys at a
 * time. Every kernel is a {@link KernelContext} kernel
 * launched with an explicit worker grid; element-wise ones map {@code ctx.globalIdx} to one item and
 * bounds-check it.
 */
public final class Chronos2Kernels {

    /** Queries per work-group of the attention kernels. */
    public static final int ATTN_QUERIES = 16;
    /** Threads per work-group of the attention kernels. */
    public static final int ATTN_THREADS = 128;
    /** Threads that share one query's dot product. */
    static final int LANES = ATTN_THREADS / ATTN_QUERIES;
    /** The widest attention head the group attention supports. */
    public static final int MAX_HEAD_DIM = 128;
    /** Queries and keys per block of {@link #timeAttention}. */
    public static final int TIME_QUERIES = 32;
    public static final int KEY_BLOCK = 32;
    /** The widest attention head {@link #timeAttention} supports (its local memory fits Metal's 32 KB). */
    public static final int BLOCK_HEAD_DIM = 64;
    /** Pitches of {@link #timeAttention}'s staged vectors and scores: one padding element, so lanes hit distinct banks. */
    static final int HEAD_PITCH = BLOCK_HEAD_DIM + 1;
    static final int SCORE_PITCH = KEY_BLOCK + 1;
    /** A masked score, and the threshold above which a score is a key's. */
    static final float MASKED = -3.4e38f;
    static final float VALID = -3.0e38f;

    private Chronos2Kernels() {}

    /**
     * {@code x [(r·n + t)·d + j]}: the patch embedding {@code emb [(r·ne + p)·d + j]} of the context
     * tokens {@code t < nc} and the future tokens {@code t ≥ nc + reg}, and the {@code [REG]} token
     * (at {@code regOff} in the packed weights) between them.
     */
    public static void assembleTokens(KernelContext ctx, FloatArray emb, FloatArray w, int regOff, FloatArray x, int rows,
            int nc, int reg, int n, int d) {
        int id = ctx.globalIdx;
        if (id < rows * n * d) {
            int j = id % d;
            int tok = id / d;
            int t = tok % n;
            int r = tok / n;
            int ne = n - reg;
            float v;
            if (t < nc) {
                v = emb.get((r * ne + t) * d + j);
            } else if (t < nc + reg) {
                v = w.get(regOff + j);
            } else {
                v = emb.get((r * ne + t - reg) * d + j);
            }
            x.set(id, v);
        }
    }

    /**
     * RoPE on q and k in place: per token and head, dimensions {@code e} and {@code e + dh/2} rotate
     * together, with {@code cos, sin} read from {@code table [t·dh + e]} and {@code [.. + dh/2 + e]},
     * {@code t} the token's index in its row (the host builds the table, as the reference rounds
     * it). One thread per (token, head, {@code e < dh/2}).
     */
    public static void rope(KernelContext ctx, FloatArray qkv, FloatArray table, int tokens, int n, int heads, int dh,
            int inner) {
        int half = dh / 2;
        int id = ctx.globalIdx;
        if (id < tokens * heads * half) {
            int e = id % half;
            int th = id / half;
            int hh = th % heads;
            int tok = th / heads;
            int p = tok % n;
            float c = table.get(p * dh + e);
            float s = table.get(p * dh + half + e);
            int qb = tok * 3 * inner + hh * dh;
            float a = qkv.get(qb + e);
            float b = qkv.get(qb + half + e);
            qkv.set(qb + e, a * c + -b * s);
            qkv.set(qb + half + e, b * c + a * s);
            int kb = qb + inner;
            float ka = qkv.get(kb + e);
            float kb2 = qkv.get(kb + half + e);
            qkv.set(kb + e, ka * c + -kb2 * s);
            qkv.set(kb + half + e, kb2 * c + ka * s);
        }
    }

    /**
     * Attention along time, flash style over blocks of keys, register-tiled: one work-group of
     * {@link #ATTN_THREADS} threads per (row, head, {@link #TIME_QUERIES} consecutive query tokens).
     * For each block of {@link #KEY_BLOCK} keys the work-group stages their k and v in local memory;
     * each thread computes a tile of 2 queries × 4 keys of scores, the 8 threads sharing those
     * queries fold the block into their online softmax together, and each thread accumulates a
     * tile of 4 queries × {@code dh / 16} output dimensions. Heads must be at most
     * {@link #BLOCK_HEAD_DIM} wide, in multiples of 16.
     *
     * <p>Every query of a row reads every valid key of the row ({@code live [r·n + k]}), in both
     * directions. Writes {@code out [(r·n + t)·inner + head·dh + j]}.
     */
    public static void timeAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray live, int rows, int n,
            int heads, int dh, int inner) {
        int wg = ctx.groupIdx;
        int tid = ctx.localIdx;
        int tiles = (n + TIME_QUERIES - 1) / TIME_QUERIES;
        int tile = wg % tiles;
        int hh = (wg / tiles) % heads;
        int r = wg / (tiles * heads);
        int q0 = tile * TIME_QUERIES;
        int qn = TornadoMath.min(TIME_QUERIES, n - q0);
        float[] qs = ctx.allocateFloatLocalArray(TIME_QUERIES * HEAD_PITCH);
        float[] ks = ctx.allocateFloatLocalArray(KEY_BLOCK * HEAD_PITCH);
        float[] vs = ctx.allocateFloatLocalArray(KEY_BLOCK * HEAD_PITCH);
        float[] sc = ctx.allocateFloatLocalArray(TIME_QUERIES * SCORE_PITCH);
        // each query's 8 partial maxima; the running maxima double-buffered by block parity
        float[] pmax = ctx.allocateFloatLocalArray(TIME_QUERIES * 8);
        float[] mRun = ctx.allocateFloatLocalArray(2 * TIME_QUERIES);
        float[] lRun = ctx.allocateFloatLocalArray(TIME_QUERIES);
        float[] corr = ctx.allocateFloatLocalArray(TIME_QUERIES);
        // the output accumulators and scores are scalars, not arrays, so every kernel compiler keeps
        // them in registers: oAD is query 4·og + A at dimension od + 16·D (D < dpt ≤ 4)
        float o00 = 0f, o01 = 0f, o02 = 0f, o03 = 0f, o10 = 0f, o11 = 0f, o12 = 0f, o13 = 0f;
        float o20 = 0f, o21 = 0f, o22 = 0f, o23 = 0f, o30 = 0f, o31 = 0f, o32 = 0f, o33 = 0f;
        for (int i = tid; i < TIME_QUERIES * dh; i += ATTN_THREADS) {
            int qq = i / dh;
            int e = i - qq * dh;
            float v = 0f;
            if (qq < qn) {
                v = qkv.get((r * n + q0 + qq) * 3 * inner + hh * dh + e);
            }
            qs[qq * HEAD_PITCH + e] = v;
        }
        if (tid < TIME_QUERIES) {
            mRun[tid] = MASKED;
            lRun[tid] = 0f;
        }
        // scores: queries 2·sg, 2·sg + 1 against keys sl, sl + 8, sl + 16, sl + 24
        int sg = tid / 8;
        int sl = tid - sg * 8;
        // output: queries 4·og .. 4·og + 3, dimensions od, od + 16, ..
        int og = tid / 16;
        int od = tid - og * 16;
        int dpt = dh / 16;
        int par = 0;
        ctx.localBarrier();
        for (int k0 = 0; k0 < n; k0 += KEY_BLOCK) {
            for (int i = tid; i < KEY_BLOCK * dh; i += ATTN_THREADS) {
                int j = i / dh;
                int e = i - j * dh;
                float kv = 0f;
                float vv = 0f;
                if (k0 + j < n) {
                    int ko = (r * n + k0 + j) * 3 * inner + hh * dh + e;
                    kv = qkv.get(ko + inner);
                    vv = qkv.get(ko + 2 * inner);
                }
                ks[j * HEAD_PITCH + e] = kv;
                vs[j * HEAD_PITCH + e] = vv;
            }
            ctx.localBarrier();
            int qa = 2 * sg * HEAD_PITCH;
            int qb = qa + HEAD_PITCH;
            float s00 = 0f, s01 = 0f, s02 = 0f, s03 = 0f;
            float s10 = 0f, s11 = 0f, s12 = 0f, s13 = 0f;
            for (int e = 0; e < dh; e++) {
                float a0 = qs[qa + e];
                float a1 = qs[qb + e];
                float b0 = ks[sl * HEAD_PITCH + e];
                float b1 = ks[(sl + 8) * HEAD_PITCH + e];
                float b2 = ks[(sl + 16) * HEAD_PITCH + e];
                float b3 = ks[(sl + 24) * HEAD_PITCH + e];
                s00 += a0 * b0;
                s01 += a0 * b1;
                s02 += a0 * b2;
                s03 += a0 * b3;
                s10 += a1 * b0;
                s11 += a1 * b1;
                s12 += a1 * b2;
                s13 += a1 * b3;
            }
            int ok0 = 0;
            if (k0 + sl + 0 < n) {
                ok0 = live.get(r * n + k0 + sl + 0);
            }
            if (ok0 == 0) {
                s00 = MASKED;
                s10 = MASKED;
            }
            int ok1 = 0;
            if (k0 + sl + 8 < n) {
                ok1 = live.get(r * n + k0 + sl + 8);
            }
            if (ok1 == 0) {
                s01 = MASKED;
                s11 = MASKED;
            }
            int ok2 = 0;
            if (k0 + sl + 16 < n) {
                ok2 = live.get(r * n + k0 + sl + 16);
            }
            if (ok2 == 0) {
                s02 = MASKED;
                s12 = MASKED;
            }
            int ok3 = 0;
            if (k0 + sl + 24 < n) {
                ok3 = live.get(r * n + k0 + sl + 24);
            }
            if (ok3 == 0) {
                s03 = MASKED;
                s13 = MASKED;
            }
            pmax[(2 * sg + 0) * 8 + sl] = TornadoMath.max(TornadoMath.max(s00, s01), TornadoMath.max(s02, s03));
            pmax[(2 * sg + 1) * 8 + sl] = TornadoMath.max(TornadoMath.max(s10, s11), TornadoMath.max(s12, s13));
            ctx.localBarrier();
            {
                int q = 2 * sg + 0;
                float bmax = MASKED;
                for (int l = 0; l < 8; l++) {
                    bmax = TornadoMath.max(bmax, pmax[q * 8 + l]);
                }
                float m = mRun[par * TIME_QUERIES + q];
                float mn = TornadoMath.max(m, bmax);
                float c = 1f;
                if (bmax > VALID) {
                    c = TornadoMath.exp(m - mn);
                }
                float pv0 = 0f;
                if (s00 > VALID) {
                    pv0 = TornadoMath.exp(s00 - mn);
                }
                sc[q * SCORE_PITCH + sl + 0] = pv0;
                float pv1 = 0f;
                if (s01 > VALID) {
                    pv1 = TornadoMath.exp(s01 - mn);
                }
                sc[q * SCORE_PITCH + sl + 8] = pv1;
                float pv2 = 0f;
                if (s02 > VALID) {
                    pv2 = TornadoMath.exp(s02 - mn);
                }
                sc[q * SCORE_PITCH + sl + 16] = pv2;
                float pv3 = 0f;
                if (s03 > VALID) {
                    pv3 = TornadoMath.exp(s03 - mn);
                }
                sc[q * SCORE_PITCH + sl + 24] = pv3;
                if (sl == 0) {
                    corr[q] = c;
                    mRun[(1 - par) * TIME_QUERIES + q] = mn;
                }
            }
            {
                int q = 2 * sg + 1;
                float bmax = MASKED;
                for (int l = 0; l < 8; l++) {
                    bmax = TornadoMath.max(bmax, pmax[q * 8 + l]);
                }
                float m = mRun[par * TIME_QUERIES + q];
                float mn = TornadoMath.max(m, bmax);
                float c = 1f;
                if (bmax > VALID) {
                    c = TornadoMath.exp(m - mn);
                }
                float pv0 = 0f;
                if (s10 > VALID) {
                    pv0 = TornadoMath.exp(s10 - mn);
                }
                sc[q * SCORE_PITCH + sl + 0] = pv0;
                float pv1 = 0f;
                if (s11 > VALID) {
                    pv1 = TornadoMath.exp(s11 - mn);
                }
                sc[q * SCORE_PITCH + sl + 8] = pv1;
                float pv2 = 0f;
                if (s12 > VALID) {
                    pv2 = TornadoMath.exp(s12 - mn);
                }
                sc[q * SCORE_PITCH + sl + 16] = pv2;
                float pv3 = 0f;
                if (s13 > VALID) {
                    pv3 = TornadoMath.exp(s13 - mn);
                }
                sc[q * SCORE_PITCH + sl + 24] = pv3;
                if (sl == 0) {
                    corr[q] = c;
                    mRun[(1 - par) * TIME_QUERIES + q] = mn;
                }
            }
            ctx.localBarrier();
            float c0 = corr[4 * og + 0];
            float c1 = corr[4 * og + 1];
            float c2 = corr[4 * og + 2];
            float c3 = corr[4 * og + 3];
            o00 = o00 * c0;
            if (1 < dpt) {
                o01 = o01 * c0;
            }
            if (2 < dpt) {
                o02 = o02 * c0;
            }
            if (3 < dpt) {
                o03 = o03 * c0;
            }
            o10 = o10 * c1;
            if (1 < dpt) {
                o11 = o11 * c1;
            }
            if (2 < dpt) {
                o12 = o12 * c1;
            }
            if (3 < dpt) {
                o13 = o13 * c1;
            }
            o20 = o20 * c2;
            if (1 < dpt) {
                o21 = o21 * c2;
            }
            if (2 < dpt) {
                o22 = o22 * c2;
            }
            if (3 < dpt) {
                o23 = o23 * c2;
            }
            o30 = o30 * c3;
            if (1 < dpt) {
                o31 = o31 * c3;
            }
            if (2 < dpt) {
                o32 = o32 * c3;
            }
            if (3 < dpt) {
                o33 = o33 * c3;
            }
            for (int j = 0; j < KEY_BLOCK; j++) {
                float p0 = sc[(4 * og) * SCORE_PITCH + j];
                float p1 = sc[(4 * og + 1) * SCORE_PITCH + j];
                float p2 = sc[(4 * og + 2) * SCORE_PITCH + j];
                float p3 = sc[(4 * og + 3) * SCORE_PITCH + j];
                float vj0 = vs[j * HEAD_PITCH + od];
                o00 += p0 * vj0;
                o10 += p1 * vj0;
                o20 += p2 * vj0;
                o30 += p3 * vj0;
                if (1 < dpt) {
                    float vj1 = vs[j * HEAD_PITCH + od + 16];
                    o01 += p0 * vj1;
                    o11 += p1 * vj1;
                    o21 += p2 * vj1;
                    o31 += p3 * vj1;
                }
                if (2 < dpt) {
                    float vj2 = vs[j * HEAD_PITCH + od + 32];
                    o02 += p0 * vj2;
                    o12 += p1 * vj2;
                    o22 += p2 * vj2;
                    o32 += p3 * vj2;
                }
                if (3 < dpt) {
                    float vj3 = vs[j * HEAD_PITCH + od + 48];
                    o03 += p0 * vj3;
                    o13 += p1 * vj3;
                    o23 += p2 * vj3;
                    o33 += p3 * vj3;
                }
            }
            if (tid < TIME_QUERIES) {
                float bsum = 0f;
                for (int j = 0; j < KEY_BLOCK; j++) {
                    bsum += sc[tid * SCORE_PITCH + j];
                }
                lRun[tid] = lRun[tid] * corr[tid] + bsum;
            }
            par = 1 - par;
            ctx.localBarrier();
        }
        if (4 * og + 0 < qn) {
            int ob = (r * n + q0 + 4 * og + 0) * inner + hh * dh + od;
            float l0 = lRun[4 * og + 0];
            out.set(ob + 0, l0 > 0f ? o00 / l0 : 0f);
            if (1 < dpt) {
                out.set(ob + 16, l0 > 0f ? o01 / l0 : 0f);
            }
            if (2 < dpt) {
                out.set(ob + 32, l0 > 0f ? o02 / l0 : 0f);
            }
            if (3 < dpt) {
                out.set(ob + 48, l0 > 0f ? o03 / l0 : 0f);
            }
        }
        if (4 * og + 1 < qn) {
            int ob = (r * n + q0 + 4 * og + 1) * inner + hh * dh + od;
            float l1 = lRun[4 * og + 1];
            out.set(ob + 0, l1 > 0f ? o10 / l1 : 0f);
            if (1 < dpt) {
                out.set(ob + 16, l1 > 0f ? o11 / l1 : 0f);
            }
            if (2 < dpt) {
                out.set(ob + 32, l1 > 0f ? o12 / l1 : 0f);
            }
            if (3 < dpt) {
                out.set(ob + 48, l1 > 0f ? o13 / l1 : 0f);
            }
        }
        if (4 * og + 2 < qn) {
            int ob = (r * n + q0 + 4 * og + 2) * inner + hh * dh + od;
            float l2 = lRun[4 * og + 2];
            out.set(ob + 0, l2 > 0f ? o20 / l2 : 0f);
            if (1 < dpt) {
                out.set(ob + 16, l2 > 0f ? o21 / l2 : 0f);
            }
            if (2 < dpt) {
                out.set(ob + 32, l2 > 0f ? o22 / l2 : 0f);
            }
            if (3 < dpt) {
                out.set(ob + 48, l2 > 0f ? o23 / l2 : 0f);
            }
        }
        if (4 * og + 3 < qn) {
            int ob = (r * n + q0 + 4 * og + 3) * inner + hh * dh + od;
            float l3 = lRun[4 * og + 3];
            out.set(ob + 0, l3 > 0f ? o30 / l3 : 0f);
            if (1 < dpt) {
                out.set(ob + 16, l3 > 0f ? o31 / l3 : 0f);
            }
            if (2 < dpt) {
                out.set(ob + 32, l3 > 0f ? o32 / l3 : 0f);
            }
            if (3 < dpt) {
                out.set(ob + 48, l3 > 0f ? o33 / l3 : 0f);
            }
        }
    }

    /**
     * Attention across the rows of each group, per token slot and head (flash style): one
     * work-group per ({@link #ATTN_QUERIES} consecutive rows, slot, head), with keys over the union of
     * the rows' groups ({@code groupStart .. groupEnd}, which must be contiguous runs of rows). A row
     * reads the rows of its group whose token at the slot is a valid key; a row with none (a context
     * patch its whole group misses) gets zeros.
     */
    public static void groupAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray live, IntArray groupStart,
            IntArray groupEnd, int rows, int n, int heads, int dh, int inner) {
        int wg = ctx.groupIdx;
        int tid = ctx.localIdx;
        int tiles = (rows + ATTN_QUERIES - 1) / ATTN_QUERIES;
        int tile = wg % tiles;
        int hh = (wg / tiles) % heads;
        int t = wg / (tiles * heads);
        int q0 = tile * ATTN_QUERIES;
        int qn = TornadoMath.min(ATTN_QUERIES, rows - q0);
        float[] qs = ctx.allocateFloatLocalArray(ATTN_QUERIES * MAX_HEAD_DIM);
        float[] kv = ctx.allocateFloatLocalArray(2 * MAX_HEAD_DIM);
        float[] part = ctx.allocateFloatLocalArray(ATTN_THREADS);
        float[] mRun = ctx.allocateFloatLocalArray(ATTN_QUERIES);
        float[] lRun = ctx.allocateFloatLocalArray(ATTN_QUERIES);
        float[] corr = ctx.allocateFloatLocalArray(ATTN_QUERIES);
        float[] pw = ctx.allocateFloatLocalArray(ATTN_QUERIES);
        // one scalar per query of ATTN_QUERIES = 16 (not an array), so every kernel compiler keeps them in registers
        float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f, a4 = 0f, a5 = 0f, a6 = 0f, a7 = 0f;
        float a8 = 0f, a9 = 0f, a10 = 0f, a11 = 0f, a12 = 0f, a13 = 0f, a14 = 0f, a15 = 0f;
        for (int qq = 0; qq < ATTN_QUERIES; qq++) {
            if (qq < qn) {
                for (int e = tid; e < dh; e += ATTN_THREADS) {
                    qs[qq * MAX_HEAD_DIM + e] = qkv.get(((q0 + qq) * n + t) * 3 * inner + hh * dh + e);
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
                int ko = (k * n + t) * 3 * inner + hh * dh + tid;
                kv[tid] = qkv.get(ko + inner);
                kv[MAX_HEAD_DIM + tid] = qkv.get(ko + 2 * inner);
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
                // int masks, not short-circuit conditions (works around a TornadoVM codegen issue)
                int inGroup = (k >= groupStart.get(qi) ? 1 : 0) * (k < groupEnd.get(qi) ? 1 : 0);
                int allowed = inGroup * live.get(k * n + t);
                if (allowed != 0) {
                    float mn = TornadoMath.max(mRun[q], dot);
                    float c = TornadoMath.exp(mRun[q] - mn);
                    float e = TornadoMath.exp(dot - mn);
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
                float v = 0f;
                if (lRun[0] > 0f) {
                    v = a0 / lRun[0];
                }
                out.set(((q0 + 0) * n + t) * inner + hh * dh + tid, v);
            }
            if (1 < qn) {
                float v = 0f;
                if (lRun[1] > 0f) {
                    v = a1 / lRun[1];
                }
                out.set(((q0 + 1) * n + t) * inner + hh * dh + tid, v);
            }
            if (2 < qn) {
                float v = 0f;
                if (lRun[2] > 0f) {
                    v = a2 / lRun[2];
                }
                out.set(((q0 + 2) * n + t) * inner + hh * dh + tid, v);
            }
            if (3 < qn) {
                float v = 0f;
                if (lRun[3] > 0f) {
                    v = a3 / lRun[3];
                }
                out.set(((q0 + 3) * n + t) * inner + hh * dh + tid, v);
            }
            if (4 < qn) {
                float v = 0f;
                if (lRun[4] > 0f) {
                    v = a4 / lRun[4];
                }
                out.set(((q0 + 4) * n + t) * inner + hh * dh + tid, v);
            }
            if (5 < qn) {
                float v = 0f;
                if (lRun[5] > 0f) {
                    v = a5 / lRun[5];
                }
                out.set(((q0 + 5) * n + t) * inner + hh * dh + tid, v);
            }
            if (6 < qn) {
                float v = 0f;
                if (lRun[6] > 0f) {
                    v = a6 / lRun[6];
                }
                out.set(((q0 + 6) * n + t) * inner + hh * dh + tid, v);
            }
            if (7 < qn) {
                float v = 0f;
                if (lRun[7] > 0f) {
                    v = a7 / lRun[7];
                }
                out.set(((q0 + 7) * n + t) * inner + hh * dh + tid, v);
            }
            if (8 < qn) {
                float v = 0f;
                if (lRun[8] > 0f) {
                    v = a8 / lRun[8];
                }
                out.set(((q0 + 8) * n + t) * inner + hh * dh + tid, v);
            }
            if (9 < qn) {
                float v = 0f;
                if (lRun[9] > 0f) {
                    v = a9 / lRun[9];
                }
                out.set(((q0 + 9) * n + t) * inner + hh * dh + tid, v);
            }
            if (10 < qn) {
                float v = 0f;
                if (lRun[10] > 0f) {
                    v = a10 / lRun[10];
                }
                out.set(((q0 + 10) * n + t) * inner + hh * dh + tid, v);
            }
            if (11 < qn) {
                float v = 0f;
                if (lRun[11] > 0f) {
                    v = a11 / lRun[11];
                }
                out.set(((q0 + 11) * n + t) * inner + hh * dh + tid, v);
            }
            if (12 < qn) {
                float v = 0f;
                if (lRun[12] > 0f) {
                    v = a12 / lRun[12];
                }
                out.set(((q0 + 12) * n + t) * inner + hh * dh + tid, v);
            }
            if (13 < qn) {
                float v = 0f;
                if (lRun[13] > 0f) {
                    v = a13 / lRun[13];
                }
                out.set(((q0 + 13) * n + t) * inner + hh * dh + tid, v);
            }
            if (14 < qn) {
                float v = 0f;
                if (lRun[14] > 0f) {
                    v = a14 / lRun[14];
                }
                out.set(((q0 + 14) * n + t) * inner + hh * dh + tid, v);
            }
            if (15 < qn) {
                float v = 0f;
                if (lRun[15] > 0f) {
                    v = a15 / lRun[15];
                }
                out.set(((q0 + 15) * n + t) * inner + hh * dh + tid, v);
            }
        }
    }

    /** {@code dec [(r·h + i)·d + j] = y [(r·n + fs + i)·d + j]}: the future tokens, for the output patch embedding. */
    public static void gatherFuture(KernelContext ctx, FloatArray y, FloatArray dec, int rows, int fs, int n, int h, int d) {
        int id = ctx.globalIdx;
        if (id < rows * h * d) {
            int j = id % d;
            int tok = id / d;
            int i = tok % h;
            int r = tok / h;
            dec.set(id, y.get((r * n + fs + i) * d + j));
        }
    }
}
