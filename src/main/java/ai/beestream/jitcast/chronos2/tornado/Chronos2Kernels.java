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
 * the {@code 1/√d_kv} scaling, as the reference. Every kernel is a {@link KernelContext} kernel
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
    /** The widest attention head the kernels support. */
    public static final int MAX_HEAD_DIM = 128;

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
     * Attention along time, flash style: one work-group of {@link #ATTN_THREADS} threads per
     * (row, head, {@link #ATTN_QUERIES} consecutive query tokens). Each key's k and v are loaded once
     * into local memory for every query of the block; {@link #LANES} threads form one query's dot
     * product, its first lane applies the mask and the online softmax, and thread {@code j < dh}
     * accumulates output dimension {@code j} of every query.
     *
     * <p>Every query of a row reads every valid key of the row ({@code live [r·n + k]}), in both
     * directions. Writes {@code out [(r·n + t)·inner + head·dh + j]}.
     */
    public static void timeAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray live, int rows, int n,
            int heads, int dh, int inner) {
        int wg = ctx.groupIdx;
        int tid = ctx.localIdx;
        int tiles = (n + ATTN_QUERIES - 1) / ATTN_QUERIES;
        int tile = wg % tiles;
        int hh = (wg / tiles) % heads;
        int r = wg / (tiles * heads);
        int q0 = tile * ATTN_QUERIES;
        int qn = TornadoMath.min(ATTN_QUERIES, n - q0);
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
                    qs[qq * MAX_HEAD_DIM + e] = qkv.get((r * n + q0 + qq) * 3 * inner + hh * dh + e);
                }
            }
        }
        if (tid < ATTN_QUERIES) {
            mRun[tid] = -3.4e38f;
            lRun[tid] = 0f;
        }
        int q = tid / LANES;
        int lane = tid - q * LANES;
        int per = dh / LANES;
        ctx.localBarrier();
        for (int k = 0; k < n; k++) {
            if (tid < dh) {
                int ko = (r * n + k) * 3 * inner + hh * dh + tid;
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
                if (live.get(r * n + k) != 0) {
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
                    out.set((r * n + q0 + qq) * inner + hh * dh + tid, acc[qq] / lRun[qq]);
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
