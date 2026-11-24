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
package ai.beestream.jitcast.timesfm.tornado;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * TimesFM 2.5's own device kernels; its GEMMs, RMSNorms and SiLU epilogues are the shared ones
 * ({@link ai.beestream.jitcast.tornado.Kernels}). Activations are row-major {@code [rows · patches, …]};
 * {@code masked[r]} is the number of padding patches that lead row {@code r}.
 */
public final class TimesFmKernels {

    private TimesFmKernels() {}

    /** Work-group width of {@link #ropeNorm}: one thread per rotated pair of q, then of k. */
    public static final int ROPE_THREADS = 128;

    /** Work-group width of {@link #causalAttention}. */
    public static final int ATTN_THREADS = 128;

    /** Queries per work-group of {@link #causalAttention}. */
    public static final int ATTN_QUERIES = 16;

    /** Keys staged in local memory at a time by {@link #causalAttention}. */
    public static final int ATTN_KEYS = 16;

    /** The widest attention head {@link #causalAttention} and {@link #ropeNorm} take. */
    public static final int ATTN_HEAD_DIM = 80;

    /** The longest sequence the engine takes, in patches (16,384 steps of 32). */
    public static final int ATTN_MAX_KEYS = 512;

    /**
     * In place on the {@code q|k|v} projections ({@code [tokens, 3·d]}): RoPE on q and k at each patch's
     * position (its index plus {@code off[0]}, the patches already decoded, minus the row's leading padding),
     * rotating each head's first half against its
     * second; then RMSNorm of each head's q and k with their scales; then q times the per-dimension
     * query scale. One work-group of {@link #ROPE_THREADS} per (token, head): thread {@code i < hd/2}
     * rotates pair {@code i} of q, thread {@code hd/2 + i} pair {@code i} of k.
     */
    public static void ropeNorm(KernelContext ctx, FloatArray qkv, FloatArray w, int qNorm, int kNorm, int qScale, int invTs,
            IntArray masked, IntArray off, int rows, int n, int heads, int hd, int d, float eps) {
        if (ctx.groupIdx < rows * n * heads) {
            int th = ctx.groupIdx;
            int head = th % heads;
            int tok = th / heads;
            int r = tok / n;
            int tid = ctx.localIdx;
            int half = hd / 2;
            float[] part = ctx.allocateFloatLocalArray(ROPE_THREADS);
            float[] sums = ctx.allocateFloatLocalArray(2);
            boolean active = tid < hd;
            int which = tid < half ? 0 : 1;
            int i = tid - which * half;
            int base = tok * 3 * d + which * d + head * hd;
            float na = 0f;
            float nb = 0f;
            if (active) {
                float pos = (float) (tok - r * n + off.get(0) - masked.get(r));
                float ang = pos * w.get(invTs + i);
                float c = TornadoMath.cos(ang);
                float s = TornadoMath.sin(ang);
                float a = qkv.get(base + i);
                float b = qkv.get(base + half + i);
                na = a * c - b * s;
                nb = b * c + a * s;
                part[tid] = na * na + nb * nb;
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
                float va = na * inv * w.get(sOff + i);
                float vb = nb * inv * w.get(sOff + half + i);
                if (which == 0) {
                    va = va * w.get(qScale + i);
                    vb = vb * w.get(qScale + half + i);
                }
                qkv.set(base + i, va);
                qkv.set(base + half + i, vb);
            }
        }
    }

    /**
     * Causal attention per (row, head) over the row's observed patches, with the queries already scaled
     * (no {@code 1/√d}); a query in the row's leading padding gets zeros. Flash-style: one work-group of
     * {@link #ATTN_THREADS} per (row, head, {@link #ATTN_QUERIES} queries) walks the keys in blocks of
     * {@link #ATTN_KEYS} staged in local memory, with an online softmax; the queries, scores and output
     * accumulators also sit in local memory.
     */
    public static void causalAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray masked, int rows, int n,
            int heads, int hd, int d) {
        int tiles = (n + ATTN_QUERIES - 1) / ATTN_QUERIES;
        if (ctx.groupIdx < rows * heads * tiles) {
            int tile = ctx.groupIdx % tiles;
            int rh = ctx.groupIdx / tiles;
            int head = rh % heads;
            int r = rh / heads;
            int tid = ctx.localIdx;
            float[] qs = ctx.allocateFloatLocalArray(ATTN_QUERIES * ATTN_HEAD_DIM);
            float[] ks = ctx.allocateFloatLocalArray(ATTN_KEYS * ATTN_HEAD_DIM);
            float[] vs = ctx.allocateFloatLocalArray(ATTN_KEYS * ATTN_HEAD_DIM);
            float[] sc = ctx.allocateFloatLocalArray(ATTN_QUERIES * ATTN_KEYS);
            float[] acc = ctx.allocateFloatLocalArray(ATTN_QUERIES * ATTN_HEAD_DIM);
            float[] mx = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] den = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] alpha = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            int m = masked.get(r);
            int q0 = tile * ATTN_QUERIES;
            int qEnd = TornadoMath.min(q0 + ATTN_QUERIES, n);
            int row0 = r * n;
            int qkvRow = 3 * d;
            for (int e = tid; e < ATTN_QUERIES * hd; e += ATTN_THREADS) {
                int qi = e / hd;
                int t = q0 + qi;
                qs[qi * ATTN_HEAD_DIM + e - qi * hd] = t < n ? qkv.get((row0 + t) * qkvRow + head * hd + e - qi * hd) : 0f;
                acc[qi * ATTN_HEAD_DIM + e - qi * hd] = 0f;
            }
            if (tid < ATTN_QUERIES) {
                mx[tid] = -3.0e38f;
                den[tid] = 0f;
            }
            ctx.localBarrier();
            for (int k0 = m; k0 < qEnd; k0 += ATTN_KEYS) {
                for (int e = tid; e < ATTN_KEYS * hd; e += ATTN_THREADS) {
                    int kj = e / hd;
                    int dd = e - kj * hd;
                    int key = k0 + kj;
                    boolean ok = key < n;
                    ks[kj * ATTN_HEAD_DIM + dd] = ok ? qkv.get((row0 + key) * qkvRow + d + head * hd + dd) : 0f;
                    vs[kj * ATTN_HEAD_DIM + dd] = ok ? qkv.get((row0 + key) * qkvRow + 2 * d + head * hd + dd) : 0f;
                }
                ctx.localBarrier();
                for (int e = tid; e < ATTN_QUERIES * ATTN_KEYS; e += ATTN_THREADS) {
                    int qi = e / ATTN_KEYS;
                    int kj = e - qi * ATTN_KEYS;
                    int t = q0 + qi;
                    int key = k0 + kj;
                    float s = -3.0e38f;
                    if (t < n && t >= m && key <= t) {
                        float dot = 0f;
                        for (int dd = 0; dd < hd; dd++) {
                            dot += qs[qi * ATTN_HEAD_DIM + dd] * ks[kj * ATTN_HEAD_DIM + dd];
                        }
                        s = dot;
                    }
                    sc[e] = s;
                }
                ctx.localBarrier();
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
                    float o = acc[qi * ATTN_HEAD_DIM + dd] * alpha[qi];
                    for (int kj = 0; kj < ATTN_KEYS; kj++) {
                        o += sc[qi * ATTN_KEYS + kj] * vs[kj * ATTN_HEAD_DIM + dd];
                    }
                    acc[qi * ATTN_HEAD_DIM + dd] = o;
                }
                ctx.localBarrier();
            }
            for (int e = tid; e < ATTN_QUERIES * hd; e += ATTN_THREADS) {
                int qi = e / hd;
                int dd = e - qi * hd;
                int t = q0 + qi;
                if (t < n) {
                    float v = t >= m ? acc[qi * ATTN_HEAD_DIM + dd] / den[qi] : 0f;
                    out.set((row0 + t) * d + head * hd + dd, v);
                }
            }
        }
    }

    /**
     * Writes the keys and values of {@code n} new patches per row ({@code [rows · n, 3·d]}) into layer
     * {@code layer} of the caches ({@code [layers, rows, cap, d]}) at patch {@code off[0]}.
     */
    public static void cacheWrite(KernelContext ctx, FloatArray qkv, FloatArray kc, FloatArray vc, IntArray off, int layer,
            int rows, int n, int cap, int d) {
        int id = ctx.globalIdx;
        if (id < rows * n * d) {
            int c = id % d;
            int tok = id / d;
            int r = tok / n;
            int j = tok - r * n;
            int dst = ((layer * rows + r) * cap + off.get(0) + j) * d + c;
            kc.set(dst, qkv.get(tok * 3 * d + d + c));
            vc.set(dst, qkv.get(tok * 3 * d + 2 * d + c));
        }
    }

    /**
     * {@link #causalAttention} for {@code n ≤ }{@link #ATTN_QUERIES} new patches per row, whose keys and
     * values (and those of every earlier patch) are in layer {@code layer} of the caches: query {@code j}
     * sits at position {@code off[0] + j} and attends to the cached patches from the row's leading padding
     * up to itself. One work-group of {@link #ATTN_THREADS} per (row, head).
     */
    public static void cachedAttention(KernelContext ctx, FloatArray qkv, FloatArray kc, FloatArray vc, FloatArray out,
            IntArray masked, IntArray off, int layer, int rows, int n, int cap, int heads, int hd, int d) {
        if (ctx.groupIdx < rows * heads) {
            int head = ctx.groupIdx % heads;
            int r = ctx.groupIdx / heads;
            int tid = ctx.localIdx;
            float[] qs = ctx.allocateFloatLocalArray(ATTN_QUERIES * ATTN_HEAD_DIM);
            float[] ks = ctx.allocateFloatLocalArray(ATTN_KEYS * ATTN_HEAD_DIM);
            float[] vs = ctx.allocateFloatLocalArray(ATTN_KEYS * ATTN_HEAD_DIM);
            float[] sc = ctx.allocateFloatLocalArray(ATTN_QUERIES * ATTN_KEYS);
            float[] acc = ctx.allocateFloatLocalArray(ATTN_QUERIES * ATTN_HEAD_DIM);
            float[] mx = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] den = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            float[] alpha = ctx.allocateFloatLocalArray(ATTN_QUERIES);
            int m = masked.get(r);
            int start = off.get(0);
            int cacheRow = (layer * rows + r) * cap;
            for (int e = tid; e < ATTN_QUERIES * hd; e += ATTN_THREADS) {
                int qi = e / hd;
                int dd = e - qi * hd;
                qs[qi * ATTN_HEAD_DIM + dd] = qi < n ? qkv.get((r * n + qi) * 3 * d + head * hd + dd) : 0f;
                acc[qi * ATTN_HEAD_DIM + dd] = 0f;
            }
            if (tid < ATTN_QUERIES) {
                mx[tid] = -3.0e38f;
                den[tid] = 0f;
            }
            ctx.localBarrier();
            int last = start + n;
            for (int k0 = m; k0 < last; k0 += ATTN_KEYS) {
                for (int e = tid; e < ATTN_KEYS * hd; e += ATTN_THREADS) {
                    int kj = e / hd;
                    int dd = e - kj * hd;
                    int key = k0 + kj;
                    boolean ok = key < last;
                    ks[kj * ATTN_HEAD_DIM + dd] = ok ? kc.get((cacheRow + key) * d + head * hd + dd) : 0f;
                    vs[kj * ATTN_HEAD_DIM + dd] = ok ? vc.get((cacheRow + key) * d + head * hd + dd) : 0f;
                }
                ctx.localBarrier();
                for (int e = tid; e < ATTN_QUERIES * ATTN_KEYS; e += ATTN_THREADS) {
                    int qi = e / ATTN_KEYS;
                    int kj = e - qi * ATTN_KEYS;
                    int key = k0 + kj;
                    float s = -3.0e38f;
                    if (qi < n && key <= start + qi) {
                        float dot = 0f;
                        for (int dd = 0; dd < hd; dd++) {
                            dot += qs[qi * ATTN_HEAD_DIM + dd] * ks[kj * ATTN_HEAD_DIM + dd];
                        }
                        s = dot;
                    }
                    sc[e] = s;
                }
                ctx.localBarrier();
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
                    float o = acc[qi * ATTN_HEAD_DIM + dd] * alpha[qi];
                    for (int kj = 0; kj < ATTN_KEYS; kj++) {
                        o += sc[qi * ATTN_KEYS + kj] * vs[kj * ATTN_HEAD_DIM + dd];
                    }
                    acc[qi * ATTN_HEAD_DIM + dd] = o;
                }
                ctx.localBarrier();
            }
            for (int e = tid; e < n * hd; e += ATTN_THREADS) {
                int qi = e / hd;
                int dd = e - qi * hd;
                out.set((r * n + qi) * d + head * hd + dd, acc[qi * ATTN_HEAD_DIM + dd] / den[qi]);
            }
        }
    }

    /** {@code dec[r] = x[r, n − 1]}: each row's last patch, for the heads. */
    public static void gatherLast(KernelContext ctx, FloatArray x, FloatArray dec, int rows, int n, int d) {
        int id = ctx.globalIdx;
        if (id < rows * d) {
            int r = id / d;
            int i = id - r * d;
            dec.set(id, x.get(((r + 1) * n - 1) * d + i));
        }
    }
}
