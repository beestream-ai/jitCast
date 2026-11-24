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

    /** Work-group width of {@link #causalAttention}. */
    public static final int ATTN_THREADS = 64;

    /** The longest sequence {@link #causalAttention} takes, in patches (16,384 steps of 32). */
    public static final int ATTN_MAX_KEYS = 512;

    /**
     * In place on the {@code q|k|v} projections ({@code [tokens, 3·d]}): RoPE on q and k at each patch's
     * position (its index minus the row's leading padding), rotating each head's first half against its
     * second; then RMSNorm of each head's q and k with their scales; then q times the per-dimension
     * query scale. One work-item per (token, head, q or k).
     */
    public static void ropeNorm(KernelContext ctx, FloatArray qkv, FloatArray w, int qNorm, int kNorm, int qScale, int invTs,
            IntArray masked, int rows, int n, int heads, int hd, int d, float eps) {
        int id = ctx.globalIdx;
        if (id < rows * n * heads * 2) {
            int part = id % 2;
            int th = id / 2;
            int head = th % heads;
            int tok = th / heads;
            int r = tok / n;
            float pos = (float) (tok - r * n - masked.get(r));
            int base = tok * 3 * d + part * d + head * hd;
            int half = hd / 2;
            float ss = 0f;
            for (int i = 0; i < half; i++) {
                float ang = pos * w.get(invTs + i);
                float c = TornadoMath.cos(ang);
                float s = TornadoMath.sin(ang);
                float a = qkv.get(base + i);
                float b = qkv.get(base + half + i);
                float na = a * c - b * s;
                float nb = b * c + a * s;
                qkv.set(base + i, na);
                qkv.set(base + half + i, nb);
                ss += na * na + nb * nb;
            }
            float inv = 1f / TornadoMath.sqrt(ss / hd + eps);
            int sOff = part == 0 ? qNorm : kNorm;
            for (int i = 0; i < hd; i++) {
                float v = qkv.get(base + i) * inv * w.get(sOff + i);
                if (part == 0) {
                    v = v * w.get(qScale + i);
                }
                qkv.set(base + i, v);
            }
        }
    }

    /**
     * Causal attention per (row, head, query patch) over the row's observed patches up to the query, with
     * the queries already scaled (no {@code 1/√d}). A query in the row's leading padding gets zeros.
     * One work-group of {@link #ATTN_THREADS} per (row, head, query); the scores sit in local memory.
     */
    public static void causalAttention(KernelContext ctx, FloatArray qkv, FloatArray out, IntArray masked, int rows, int n,
            int heads, int hd, int d) {
        int gid = ctx.groupIdx;
        if (gid < rows * heads * n) {
            int t = gid % n;
            int rh = gid / n;
            int head = rh % heads;
            int r = rh / heads;
            int tid = ctx.localIdx;
            float[] s = ctx.allocateFloatLocalArray(ATTN_MAX_KEYS);
            float[] red = ctx.allocateFloatLocalArray(ATTN_THREADS);
            int m = masked.get(r);
            int ob = (r * n + t) * d + head * hd;
            if (t < m) {
                for (int i = tid; i < hd; i += ATTN_THREADS) {
                    out.set(ob + i, 0f);
                }
            } else {
                int qb = (r * n + t) * 3 * d + head * hd;
                float mx = -3.0e38f;
                for (int j = m + tid; j <= t; j += ATTN_THREADS) {
                    int kb = (r * n + j) * 3 * d + d + head * hd;
                    float acc = 0f;
                    for (int i = 0; i < hd; i++) {
                        acc += qkv.get(qb + i) * qkv.get(kb + i);
                    }
                    s[j] = acc;
                    mx = TornadoMath.max(mx, acc);
                }
                red[tid] = mx;
                ctx.localBarrier();
                for (int st = ATTN_THREADS / 2; st > 0; st >>= 1) {
                    if (tid < st) {
                        red[tid] = TornadoMath.max(red[tid], red[tid + st]);
                    }
                    ctx.localBarrier();
                }
                float gmax = red[0];
                ctx.localBarrier();
                float sum = 0f;
                for (int j = m + tid; j <= t; j += ATTN_THREADS) {
                    float e = TornadoMath.exp(s[j] - gmax);
                    s[j] = e;
                    sum += e;
                }
                red[tid] = sum;
                ctx.localBarrier();
                for (int st = ATTN_THREADS / 2; st > 0; st >>= 1) {
                    if (tid < st) {
                        red[tid] += red[tid + st];
                    }
                    ctx.localBarrier();
                }
                float total = red[0];
                for (int i = tid; i < hd; i += ATTN_THREADS) {
                    float acc = 0f;
                    for (int j = m; j <= t; j++) {
                        acc += s[j] * qkv.get((r * n + j) * 3 * d + 2 * d + head * hd + i);
                    }
                    out.set(ob + i, acc / total);
                }
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
