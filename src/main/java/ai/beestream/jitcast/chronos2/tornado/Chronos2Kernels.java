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
        float[] acc = new float[16];
        float[] sv = new float[8];
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
        for (int i = 0; i < 16; i++) {
            acc[i] = 0f;
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
            sv[0] = s00;
            sv[1] = s01;
            sv[2] = s02;
            sv[3] = s03;
            sv[4] = s10;
            sv[5] = s11;
            sv[6] = s12;
            sv[7] = s13;
            for (int i = 0; i < 4; i++) {
                int ok = 0;
                if (k0 + sl + 8 * i < n) {
                    ok = live.get(r * n + k0 + sl + 8 * i);
                }
                if (ok == 0) {
                    sv[i] = MASKED;
                    sv[4 + i] = MASKED;
                }
            }
            for (int qi = 0; qi < 2; qi++) {
                float lmax = MASKED;
                for (int i = 0; i < 4; i++) {
                    lmax = TornadoMath.max(lmax, sv[qi * 4 + i]);
                }
                pmax[(2 * sg + qi) * 8 + sl] = lmax;
            }
            ctx.localBarrier();
            for (int qi = 0; qi < 2; qi++) {
                int q = 2 * sg + qi;
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
                for (int i = 0; i < 4; i++) {
                    float pv = 0f;
                    if (sv[qi * 4 + i] > VALID) {
                        pv = TornadoMath.exp(sv[qi * 4 + i] - mn);
                    }
                    sc[q * SCORE_PITCH + sl + 8 * i] = pv;
                }
                if (sl == 0) {
                    corr[q] = c;
                    mRun[(1 - par) * TIME_QUERIES + q] = mn;
                }
            }
            ctx.localBarrier();
            for (int a = 0; a < 4; a++) {
                float c = corr[4 * og + a];
                for (int d = 0; d < dpt; d++) {
                    acc[a * 4 + d] = acc[a * 4 + d] * c;
                }
            }
            for (int j = 0; j < KEY_BLOCK; j++) {
                float p0 = sc[(4 * og) * SCORE_PITCH + j];
                float p1 = sc[(4 * og + 1) * SCORE_PITCH + j];
                float p2 = sc[(4 * og + 2) * SCORE_PITCH + j];
                float p3 = sc[(4 * og + 3) * SCORE_PITCH + j];
                for (int d = 0; d < dpt; d++) {
                    float vj = vs[j * HEAD_PITCH + od + 16 * d];
                    acc[d] += p0 * vj;
                    acc[4 + d] += p1 * vj;
                    acc[8 + d] += p2 * vj;
                    acc[12 + d] += p3 * vj;
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
        for (int a = 0; a < 4; a++) {
            int qq = 4 * og + a;
            if (qq < qn) {
                float l = lRun[qq];
                for (int d = 0; d < dpt; d++) {
                    float v = 0f;
                    if (l > 0f) {
                        v = acc[a * 4 + d] / l;
                    }
                    out.set((r * n + q0 + qq) * inner + hh * dh + od + 16 * d, v);
                }
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
        float[] acc = new float[ATTN_QUERIES];
        for (int qq = 0; qq < ATTN_QUERIES; qq++) {
            acc[qq] = 0f;
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
                for (int qq = 0; qq < ATTN_QUERIES; qq++) {
                    if (qq < qn) {
                        acc[qq] = acc[qq] * corr[qq] + pw[qq] * vj;
                    }
                }
            }
        }
        if (tid < dh) {
            for (int qq = 0; qq < ATTN_QUERIES; qq++) {
                if (qq < qn) {
                    float l = lRun[qq];
                    float v = 0f;
                    if (l > 0f) {
                        v = acc[qq] / l;
                    }
                    out.set(((q0 + qq) * n + t) * inner + hh * dh + tid, v);
                }
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
