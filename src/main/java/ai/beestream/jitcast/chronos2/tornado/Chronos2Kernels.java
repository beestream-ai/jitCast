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

import ai.beestream.jitcast.tornado.Kernels;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.tile.DType;
import uk.ac.manchester.tornado.api.tile.PartitionView;
import uk.ac.manchester.tornado.api.tile.Tile;
import uk.ac.manchester.tornado.api.tile.TileContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
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
    /** Threads, queries (16 per warp), keys per block and head width of {@link #timeAttentionMma}. */
    public static final int MMA_THREADS = 128;
    public static final int MMA_QUERIES = 64;
    public static final int MMA_KEYS = 32;
    public static final int MMA_HEAD_DIM = 64;
    /** Shared-memory row pitches, in FP16 pairs, of its K and transposed V tiles: padded so a fragment read hits 32 banks. */
    static final int MMA_K_PITCH = MMA_HEAD_DIM / 2 + 4;
    static final int MMA_V_PITCH = MMA_KEYS / 2 + 4;
    /** The head width of {@link #timeAttentionTile}, a tile extent, so a compile-time constant. */
    public static final int TILE_HEAD_DIM = 64;
    /** Queries and keys per tile block of {@link #timeAttentionTile}. */
    public static final int TILE_QUERIES = 64;
    public static final int TILE_KEYS = 64;

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
     * directions. Writes {@code out [(r·n + t)·inner + head·dh + j]}, and its FP16 twin {@code th}
     * when {@code twin} is 1.
     */
    public static void timeAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray live, int rows, int n,
            int heads, int dh, int inner, HalfFloatArray th, int twin) {
        if (ctx.groupIdx < rows * heads * ((n + TIME_QUERIES - 1) / TIME_QUERIES)) {
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
                store(out, th, twin, ob + 0, l0 > 0f ? o00 / l0 : 0f);
                if (1 < dpt) {
                    store(out, th, twin, ob + 16, l0 > 0f ? o01 / l0 : 0f);
                }
                if (2 < dpt) {
                    store(out, th, twin, ob + 32, l0 > 0f ? o02 / l0 : 0f);
                }
                if (3 < dpt) {
                    store(out, th, twin, ob + 48, l0 > 0f ? o03 / l0 : 0f);
                }
            }
            if (4 * og + 1 < qn) {
                int ob = (r * n + q0 + 4 * og + 1) * inner + hh * dh + od;
                float l1 = lRun[4 * og + 1];
                store(out, th, twin, ob + 0, l1 > 0f ? o10 / l1 : 0f);
                if (1 < dpt) {
                    store(out, th, twin, ob + 16, l1 > 0f ? o11 / l1 : 0f);
                }
                if (2 < dpt) {
                    store(out, th, twin, ob + 32, l1 > 0f ? o12 / l1 : 0f);
                }
                if (3 < dpt) {
                    store(out, th, twin, ob + 48, l1 > 0f ? o13 / l1 : 0f);
                }
            }
            if (4 * og + 2 < qn) {
                int ob = (r * n + q0 + 4 * og + 2) * inner + hh * dh + od;
                float l2 = lRun[4 * og + 2];
                store(out, th, twin, ob + 0, l2 > 0f ? o20 / l2 : 0f);
                if (1 < dpt) {
                    store(out, th, twin, ob + 16, l2 > 0f ? o21 / l2 : 0f);
                }
                if (2 < dpt) {
                    store(out, th, twin, ob + 32, l2 > 0f ? o22 / l2 : 0f);
                }
                if (3 < dpt) {
                    store(out, th, twin, ob + 48, l2 > 0f ? o23 / l2 : 0f);
                }
            }
            if (4 * og + 3 < qn) {
                int ob = (r * n + q0 + 4 * og + 3) * inner + hh * dh + od;
                float l3 = lRun[4 * og + 3];
                store(out, th, twin, ob + 0, l3 > 0f ? o30 / l3 : 0f);
                if (1 < dpt) {
                    store(out, th, twin, ob + 16, l3 > 0f ? o31 / l3 : 0f);
                }
                if (2 < dpt) {
                    store(out, th, twin, ob + 32, l3 > 0f ? o32 / l3 : 0f);
                }
                if (3 < dpt) {
                    store(out, th, twin, ob + 48, l3 > 0f ? o33 / l3 : 0f);
                }
            }
        }
    }

    /**
     * {@link #timeAttention} with the CUDA Tile API (CUDA only, compute capability 8.0+): flash
     * attention on tensor cores, FP16 operands and FP32 scores, softmax and output. One tile block per
     * ({@link #TILE_QUERIES} queries, row·head): the grid is {@code (⌈n / TILE_QUERIES⌉, rows · heads)}
     * tile blocks. Q, K and V are read in place from {@code qkv} through strided views and cast to FP16
     * per block; the block of {@link #TILE_KEYS} keys is masked with {@code live}, and the scores fold
     * into an online softmax. Heads must be {@link #TILE_HEAD_DIM} wide; masked loads cover a ragged
     * {@code n}, and the keys past it read as not live. With {@code twin} 1 the output's FP16 twin goes
     * to {@code th} too.
     */
    public static void timeAttentionTile(TileContext tc, FloatArray qkv, FloatArray out, IntArray live, int n, int heads,
            int inner, int keyBlocks, HalfFloatArray th, int twin) {
        int rh = tc.bidY();
        int r = rh / heads;
        int hh = rh - r * heads;
        int base = r * n * 3 * inner + hh * TILE_HEAD_DIM;
        PartitionView qv = tc.partition(tc.viewStrided(qkv, base, n, TILE_HEAD_DIM, 3 * inner), TILE_QUERIES, TILE_HEAD_DIM);
        PartitionView kv = tc.partition(tc.viewStrided(qkv, base + inner, n, TILE_HEAD_DIM, 3 * inner), TILE_KEYS,
                TILE_HEAD_DIM);
        PartitionView vv = tc.partition(tc.viewStrided(qkv, base + 2 * inner, n, TILE_HEAD_DIM, 3 * inner), TILE_KEYS,
                TILE_HEAD_DIM);
        PartitionView lv = tc.partition(tc.viewStrided(live, r * n, 1, n, n), 1, TILE_KEYS);
        int oBase = r * n * inner + hh * TILE_HEAD_DIM;
        PartitionView ov = tc.partition(tc.viewStrided(out, oBase, n, TILE_HEAD_DIM, inner), TILE_QUERIES, TILE_HEAD_DIM);
        int qb = tc.bidX();
        Tile query = tc.cast(qv.loadMasked(qb, 0), DType.F16);
        Tile masked = tc.full(DType.F32, MASKED, TILE_QUERIES, TILE_KEYS);
        Tile rowMax = tc.full(DType.F32, VALID, TILE_QUERIES, 1);
        Tile rowSum = tc.zeros(DType.F32, TILE_QUERIES, 1);
        Tile acc = tc.zeros(DType.F32, TILE_QUERIES, TILE_HEAD_DIM);
        for (int b = 0; b < keyBlocks; b++) {
            Tile keys = tc.transpose(tc.cast(kv.loadMasked(b, 0), DType.F16));
            Tile scores = tc.mma(query, keys, tc.zeros(DType.F32, TILE_QUERIES, TILE_KEYS));
            Tile ok = tc.broadcast(tc.notEqualTo(lv.loadMasked(0, b), 0), TILE_QUERIES, TILE_KEYS);
            scores = tc.select(ok, scores, masked);
            Tile newMax = tc.maximum(rowMax, tc.max(scores, 1));
            Tile p = tc.exp(tc.sub(scores, newMax));
            Tile correction = tc.exp(tc.sub(rowMax, newMax));
            rowSum = tc.add(tc.mul(rowSum, correction), tc.sum(p, 1));
            acc = tc.mma(tc.cast(p, DType.F16), tc.cast(vv.loadMasked(b, 0), DType.F16), tc.mul(acc, correction));
            rowMax = newMax;
        }
        // every row has a live key (its [REG] and future tokens), so rowSum > 0 for every real query
        Tile o = tc.div(acc, rowSum);
        ov.storeMasked(o, qb, 0);
        if (twin == 1) {
            PartitionView hv = tc.partition(tc.viewStrided(th, oBase, n, TILE_HEAD_DIM, inner), TILE_QUERIES, TILE_HEAD_DIM);
            Tile limit = tc.full(DType.F32, Kernels.HALF_MAX, TILE_QUERIES, TILE_HEAD_DIM);
            hv.storeMasked(tc.cast(tc.minimum(tc.maximum(o, tc.scale(limit, -1.0)), limit), DType.F16), qb, 0);
        }
    }

    /**
     * {@link #timeAttention} on tensor cores with warp-level {@code mma} (CUDA only, compute capability
     * 8.0+), flash attention in the FlashAttention-2 layout: one work-group of {@link #MMA_THREADS}
     * threads per (row, head, {@link #MMA_QUERIES} consecutive query tokens), each warp 16 of the queries.
     * Each lane packs its Q fragments from {@code qkv} once; per block of {@link #MMA_KEYS} keys the
     * work-group stages K and V in shared memory as FP16 pairs (V transposed), each warp computes its
     * 16 × {@link #MMA_KEYS} scores with {@code mma}, folds them into its online softmax in registers
     * and feeds the probabilities straight back as the A fragments of P·V. The output accumulates in
     * FP32 registers. Heads must be {@link #MMA_HEAD_DIM} wide. Writes {@code out} as
     * {@link #timeAttention} does, and its FP16 twin {@code th} when {@code twin} is 1.
     */
    public static void timeAttentionMma(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray live, int rows, int n,
            int heads, int inner, HalfFloatArray th, int twin) {
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
            int qBase = r * n * 3 * inner + hh * MMA_HEAD_DIM;
            int qa = qBase + ra * 3 * inner;
            int qb = qBase + rb * 3 * inner;
            boolean okA = ra < n;
            boolean okB = rb < n;
            int q00 = pack(okA ? qkv.get(qa + 0 + c) : 0f, okA ? qkv.get(qa + 0 + c + 1) : 0f);
            int q01 = pack(okB ? qkv.get(qb + 0 + c) : 0f, okB ? qkv.get(qb + 0 + c + 1) : 0f);
            int q02 = pack(okA ? qkv.get(qa + 8 + c) : 0f, okA ? qkv.get(qa + 8 + c + 1) : 0f);
            int q03 = pack(okB ? qkv.get(qb + 8 + c) : 0f, okB ? qkv.get(qb + 8 + c + 1) : 0f);
            int q10 = pack(okA ? qkv.get(qa + 16 + c) : 0f, okA ? qkv.get(qa + 16 + c + 1) : 0f);
            int q11 = pack(okB ? qkv.get(qb + 16 + c) : 0f, okB ? qkv.get(qb + 16 + c + 1) : 0f);
            int q12 = pack(okA ? qkv.get(qa + 24 + c) : 0f, okA ? qkv.get(qa + 24 + c + 1) : 0f);
            int q13 = pack(okB ? qkv.get(qb + 24 + c) : 0f, okB ? qkv.get(qb + 24 + c + 1) : 0f);
            int q20 = pack(okA ? qkv.get(qa + 32 + c) : 0f, okA ? qkv.get(qa + 32 + c + 1) : 0f);
            int q21 = pack(okB ? qkv.get(qb + 32 + c) : 0f, okB ? qkv.get(qb + 32 + c + 1) : 0f);
            int q22 = pack(okA ? qkv.get(qa + 40 + c) : 0f, okA ? qkv.get(qa + 40 + c + 1) : 0f);
            int q23 = pack(okB ? qkv.get(qb + 40 + c) : 0f, okB ? qkv.get(qb + 40 + c + 1) : 0f);
            int q30 = pack(okA ? qkv.get(qa + 48 + c) : 0f, okA ? qkv.get(qa + 48 + c + 1) : 0f);
            int q31 = pack(okB ? qkv.get(qb + 48 + c) : 0f, okB ? qkv.get(qb + 48 + c + 1) : 0f);
            int q32 = pack(okA ? qkv.get(qa + 56 + c) : 0f, okA ? qkv.get(qa + 56 + c + 1) : 0f);
            int q33 = pack(okB ? qkv.get(qb + 56 + c) : 0f, okB ? qkv.get(qb + 56 + c + 1) : 0f);
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
            for (int k0 = 0; k0 < n; k0 += MMA_KEYS) {
                // K: key j, dimensions 2p, 2p + 1 at ks[j·pitch + p]; V transposed: dimension d, keys 2p, 2p + 1 at vs[d·pitch + p]
                for (int i = tid; i < MMA_KEYS * MMA_HEAD_DIM / 2; i += MMA_THREADS) {
                    int j = i >> 5;
                    int p = i & 31;
                    int ko = qBase + (k0 + j) * 3 * inner + inner + 2 * p;
                    boolean in = k0 + j < n;
                    ks[j * MMA_K_PITCH + p] = pack(in ? qkv.get(ko) : 0f, in ? qkv.get(ko + 1) : 0f);
                }
                for (int i = tid; i < MMA_KEYS * MMA_HEAD_DIM / 2; i += MMA_THREADS) {
                    int d = i & (MMA_HEAD_DIM - 1);
                    int p = i >> 6;
                    int vo = qBase + (k0 + 2 * p) * 3 * inner + 2 * inner + d;
                    boolean in0 = k0 + 2 * p < n;
                    boolean in1 = k0 + 2 * p + 1 < n;
                    vs[d * MMA_V_PITCH + p] = pack(in0 ? qkv.get(vo) : 0f, in1 ? qkv.get(vo + 3 * inner) : 0f);
                }
                ctx.localBarrier();
                HalfFloat[] fq0 = ctx.mmaFragmentA(q00, q01, q02, q03);
                HalfFloat[] fq1 = ctx.mmaFragmentA(q10, q11, q12, q13);
                HalfFloat[] fq2 = ctx.mmaFragmentA(q20, q21, q22, q23);
                HalfFloat[] fq3 = ctx.mmaFragmentA(q30, q31, q32, q33);
                float[] s0 = ctx.mmaFragment(0f);
                int kr0 = (0 + (lane >> 2)) * MMA_K_PITCH + quad;
                s0 = ctx.mma(fq0, ctx.mmaFragmentB(ks[kr0 + 0], ks[kr0 + 4]), s0, MMAShape.M16N8K16);
                s0 = ctx.mma(fq1, ctx.mmaFragmentB(ks[kr0 + 8], ks[kr0 + 12]), s0, MMAShape.M16N8K16);
                s0 = ctx.mma(fq2, ctx.mmaFragmentB(ks[kr0 + 16], ks[kr0 + 20]), s0, MMAShape.M16N8K16);
                s0 = ctx.mma(fq3, ctx.mmaFragmentB(ks[kr0 + 24], ks[kr0 + 28]), s0, MMAShape.M16N8K16);
                float[] s1 = ctx.mmaFragment(0f);
                int kr1 = (8 + (lane >> 2)) * MMA_K_PITCH + quad;
                s1 = ctx.mma(fq0, ctx.mmaFragmentB(ks[kr1 + 0], ks[kr1 + 4]), s1, MMAShape.M16N8K16);
                s1 = ctx.mma(fq1, ctx.mmaFragmentB(ks[kr1 + 8], ks[kr1 + 12]), s1, MMAShape.M16N8K16);
                s1 = ctx.mma(fq2, ctx.mmaFragmentB(ks[kr1 + 16], ks[kr1 + 20]), s1, MMAShape.M16N8K16);
                s1 = ctx.mma(fq3, ctx.mmaFragmentB(ks[kr1 + 24], ks[kr1 + 28]), s1, MMAShape.M16N8K16);
                float[] s2 = ctx.mmaFragment(0f);
                int kr2 = (16 + (lane >> 2)) * MMA_K_PITCH + quad;
                s2 = ctx.mma(fq0, ctx.mmaFragmentB(ks[kr2 + 0], ks[kr2 + 4]), s2, MMAShape.M16N8K16);
                s2 = ctx.mma(fq1, ctx.mmaFragmentB(ks[kr2 + 8], ks[kr2 + 12]), s2, MMAShape.M16N8K16);
                s2 = ctx.mma(fq2, ctx.mmaFragmentB(ks[kr2 + 16], ks[kr2 + 20]), s2, MMAShape.M16N8K16);
                s2 = ctx.mma(fq3, ctx.mmaFragmentB(ks[kr2 + 24], ks[kr2 + 28]), s2, MMAShape.M16N8K16);
                float[] s3 = ctx.mmaFragment(0f);
                int kr3 = (24 + (lane >> 2)) * MMA_K_PITCH + quad;
                s3 = ctx.mma(fq0, ctx.mmaFragmentB(ks[kr3 + 0], ks[kr3 + 4]), s3, MMAShape.M16N8K16);
                s3 = ctx.mma(fq1, ctx.mmaFragmentB(ks[kr3 + 8], ks[kr3 + 12]), s3, MMAShape.M16N8K16);
                s3 = ctx.mma(fq2, ctx.mmaFragmentB(ks[kr3 + 16], ks[kr3 + 20]), s3, MMAShape.M16N8K16);
                s3 = ctx.mma(fq3, ctx.mmaFragmentB(ks[kr3 + 24], ks[kr3 + 28]), s3, MMAShape.M16N8K16);
                boolean live00 = k0 + 0 + c < n && live.get(r * n + k0 + 0 + c) != 0;
                float x00 = live00 ? s0[0] : MASKED;
                float y00 = live00 ? s0[2] : MASKED;
                boolean live01 = k0 + 1 + c < n && live.get(r * n + k0 + 1 + c) != 0;
                float x01 = live01 ? s0[1] : MASKED;
                float y01 = live01 ? s0[3] : MASKED;
                boolean live10 = k0 + 8 + c < n && live.get(r * n + k0 + 8 + c) != 0;
                float x10 = live10 ? s1[0] : MASKED;
                float y10 = live10 ? s1[2] : MASKED;
                boolean live11 = k0 + 9 + c < n && live.get(r * n + k0 + 9 + c) != 0;
                float x11 = live11 ? s1[1] : MASKED;
                float y11 = live11 ? s1[3] : MASKED;
                boolean live20 = k0 + 16 + c < n && live.get(r * n + k0 + 16 + c) != 0;
                float x20 = live20 ? s2[0] : MASKED;
                float y20 = live20 ? s2[2] : MASKED;
                boolean live21 = k0 + 17 + c < n && live.get(r * n + k0 + 17 + c) != 0;
                float x21 = live21 ? s2[1] : MASKED;
                float y21 = live21 ? s2[3] : MASKED;
                boolean live30 = k0 + 24 + c < n && live.get(r * n + k0 + 24 + c) != 0;
                float x30 = live30 ? s3[0] : MASKED;
                float y30 = live30 ? s3[2] : MASKED;
                boolean live31 = k0 + 25 + c < n && live.get(r * n + k0 + 25 + c) != 0;
                float x31 = live31 ? s3[1] : MASKED;
                float y31 = live31 ? s3[3] : MASKED;
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
                o00 = o00 * cA + t0[0];
                o01 = o01 * cA + t0[1];
                o02 = o02 * cB + t0[2];
                o03 = o03 * cB + t0[3];
                int vr1 = (8 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t1 = ctx.mmaFragment(0f);
                t1 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr1 + 0], vs[vr1 + 4]), t1, MMAShape.M16N8K16);
                t1 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr1 + 8], vs[vr1 + 12]), t1, MMAShape.M16N8K16);
                o10 = o10 * cA + t1[0];
                o11 = o11 * cA + t1[1];
                o12 = o12 * cB + t1[2];
                o13 = o13 * cB + t1[3];
                int vr2 = (16 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t2 = ctx.mmaFragment(0f);
                t2 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr2 + 0], vs[vr2 + 4]), t2, MMAShape.M16N8K16);
                t2 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr2 + 8], vs[vr2 + 12]), t2, MMAShape.M16N8K16);
                o20 = o20 * cA + t2[0];
                o21 = o21 * cA + t2[1];
                o22 = o22 * cB + t2[2];
                o23 = o23 * cB + t2[3];
                int vr3 = (24 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t3 = ctx.mmaFragment(0f);
                t3 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr3 + 0], vs[vr3 + 4]), t3, MMAShape.M16N8K16);
                t3 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr3 + 8], vs[vr3 + 12]), t3, MMAShape.M16N8K16);
                o30 = o30 * cA + t3[0];
                o31 = o31 * cA + t3[1];
                o32 = o32 * cB + t3[2];
                o33 = o33 * cB + t3[3];
                int vr4 = (32 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t4 = ctx.mmaFragment(0f);
                t4 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr4 + 0], vs[vr4 + 4]), t4, MMAShape.M16N8K16);
                t4 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr4 + 8], vs[vr4 + 12]), t4, MMAShape.M16N8K16);
                o40 = o40 * cA + t4[0];
                o41 = o41 * cA + t4[1];
                o42 = o42 * cB + t4[2];
                o43 = o43 * cB + t4[3];
                int vr5 = (40 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t5 = ctx.mmaFragment(0f);
                t5 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr5 + 0], vs[vr5 + 4]), t5, MMAShape.M16N8K16);
                t5 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr5 + 8], vs[vr5 + 12]), t5, MMAShape.M16N8K16);
                o50 = o50 * cA + t5[0];
                o51 = o51 * cA + t5[1];
                o52 = o52 * cB + t5[2];
                o53 = o53 * cB + t5[3];
                int vr6 = (48 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t6 = ctx.mmaFragment(0f);
                t6 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr6 + 0], vs[vr6 + 4]), t6, MMAShape.M16N8K16);
                t6 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr6 + 8], vs[vr6 + 12]), t6, MMAShape.M16N8K16);
                o60 = o60 * cA + t6[0];
                o61 = o61 * cA + t6[1];
                o62 = o62 * cB + t6[2];
                o63 = o63 * cB + t6[3];
                int vr7 = (56 + (lane >> 2)) * MMA_V_PITCH + quad;
                float[] t7 = ctx.mmaFragment(0f);
                t7 = ctx.mma(fp0, ctx.mmaFragmentB(vs[vr7 + 0], vs[vr7 + 4]), t7, MMAShape.M16N8K16);
                t7 = ctx.mma(fp1, ctx.mmaFragmentB(vs[vr7 + 8], vs[vr7 + 12]), t7, MMAShape.M16N8K16);
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
            // every row has a live key (its [REG] and future tokens), so l > 0 for every real query
            float iA = 1f / lA;
            float iB = 1f / lB;
            int oa = (r * n + ra) * inner + hh * MMA_HEAD_DIM + c;
            int ob = (r * n + rb) * inner + hh * MMA_HEAD_DIM + c;
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

    /** {@code x} and {@code y} as FP16 in one register, {@code x} in bits 0-15: a half pair of an {@code mma} fragment. */
    static int pack(float x, float y) {
        return (new HalfFloat(x).getHalfFloatValue() & 0xFFFF) | ((new HalfFloat(y).getHalfFloatValue() & 0xFFFF) << 16);
    }

    /** {@code out[i] = v}, and its FP16 twin, saturated to ±{@code HALF_MAX}, when {@code twin} is 1. */
    static void store(FloatArray out, HalfFloatArray th, int twin, int i, float v) {
        out.set(i, v);
        if (twin == 1) {
            th.set(i, new HalfFloat(TornadoMath.clamp(v, -Kernels.HALF_MAX, Kernels.HALF_MAX)));
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
            IntArray groupEnd, int rows, int n, int heads, int dh, int inner, HalfFloatArray th, int twin) {
        if (ctx.groupIdx < n * heads * ((rows + ATTN_QUERIES - 1) / ATTN_QUERIES)) {
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
                    store(out, th, twin, ((q0 + 0) * n + t) * inner + hh * dh + tid, v);
                }
                if (1 < qn) {
                    float v = 0f;
                    if (lRun[1] > 0f) {
                        v = a1 / lRun[1];
                    }
                    store(out, th, twin, ((q0 + 1) * n + t) * inner + hh * dh + tid, v);
                }
                if (2 < qn) {
                    float v = 0f;
                    if (lRun[2] > 0f) {
                        v = a2 / lRun[2];
                    }
                    store(out, th, twin, ((q0 + 2) * n + t) * inner + hh * dh + tid, v);
                }
                if (3 < qn) {
                    float v = 0f;
                    if (lRun[3] > 0f) {
                        v = a3 / lRun[3];
                    }
                    store(out, th, twin, ((q0 + 3) * n + t) * inner + hh * dh + tid, v);
                }
                if (4 < qn) {
                    float v = 0f;
                    if (lRun[4] > 0f) {
                        v = a4 / lRun[4];
                    }
                    store(out, th, twin, ((q0 + 4) * n + t) * inner + hh * dh + tid, v);
                }
                if (5 < qn) {
                    float v = 0f;
                    if (lRun[5] > 0f) {
                        v = a5 / lRun[5];
                    }
                    store(out, th, twin, ((q0 + 5) * n + t) * inner + hh * dh + tid, v);
                }
                if (6 < qn) {
                    float v = 0f;
                    if (lRun[6] > 0f) {
                        v = a6 / lRun[6];
                    }
                    store(out, th, twin, ((q0 + 6) * n + t) * inner + hh * dh + tid, v);
                }
                if (7 < qn) {
                    float v = 0f;
                    if (lRun[7] > 0f) {
                        v = a7 / lRun[7];
                    }
                    store(out, th, twin, ((q0 + 7) * n + t) * inner + hh * dh + tid, v);
                }
                if (8 < qn) {
                    float v = 0f;
                    if (lRun[8] > 0f) {
                        v = a8 / lRun[8];
                    }
                    store(out, th, twin, ((q0 + 8) * n + t) * inner + hh * dh + tid, v);
                }
                if (9 < qn) {
                    float v = 0f;
                    if (lRun[9] > 0f) {
                        v = a9 / lRun[9];
                    }
                    store(out, th, twin, ((q0 + 9) * n + t) * inner + hh * dh + tid, v);
                }
                if (10 < qn) {
                    float v = 0f;
                    if (lRun[10] > 0f) {
                        v = a10 / lRun[10];
                    }
                    store(out, th, twin, ((q0 + 10) * n + t) * inner + hh * dh + tid, v);
                }
                if (11 < qn) {
                    float v = 0f;
                    if (lRun[11] > 0f) {
                        v = a11 / lRun[11];
                    }
                    store(out, th, twin, ((q0 + 11) * n + t) * inner + hh * dh + tid, v);
                }
                if (12 < qn) {
                    float v = 0f;
                    if (lRun[12] > 0f) {
                        v = a12 / lRun[12];
                    }
                    store(out, th, twin, ((q0 + 12) * n + t) * inner + hh * dh + tid, v);
                }
                if (13 < qn) {
                    float v = 0f;
                    if (lRun[13] > 0f) {
                        v = a13 / lRun[13];
                    }
                    store(out, th, twin, ((q0 + 13) * n + t) * inner + hh * dh + tid, v);
                }
                if (14 < qn) {
                    float v = 0f;
                    if (lRun[14] > 0f) {
                        v = a14 / lRun[14];
                    }
                    store(out, th, twin, ((q0 + 14) * n + t) * inner + hh * dh + tid, v);
                }
                if (15 < qn) {
                    float v = 0f;
                    if (lRun[15] > 0f) {
                        v = a15 / lRun[15];
                    }
                    store(out, th, twin, ((q0 + 15) * n + t) * inner + hh * dh + tid, v);
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
