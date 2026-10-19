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

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
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
        float[] acc = new float[ATTN_QUERIES];
        for (int qq = 0; qq < ATTN_QUERIES; qq++) {
            acc[qq] = 0f;
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
                    out.set((r * n + q0 + qq) * d + hh * dh + tid, acc[qq] / lRun[qq]);
                }
            }
        }
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
        float[] acc = new float[ATTN_QUERIES];
        for (int qq = 0; qq < ATTN_QUERIES; qq++) {
            acc[qq] = 0f;
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
                    out.set(((q0 + qq) * n + t) * d + hh * dh + tid, acc[qq] / lRun[qq]);
                }
            }
        }
    }

    /** {@code att[i] ·= sigmoid(g[i])} for the first {@code count} elements (the attention output gate). */
    public static void gate(KernelContext ctx, FloatArray att, FloatArray g, int count) {
        int i = ctx.globalIdx;
        if (i < count) {
            att.set(i, att.get(i) / (1f + TornadoMath.exp(-g.get(i))));
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
