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
package ai.beestream.jitcast.tirex2.tornado;

import ai.beestream.jitcast.tornado.Kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The TornadoVM kernels specific to the TiRex-2 forward pass (portable: OpenCL, Metal, CUDA): the
 * scaler, the mLSTM and sLSTM cells, the head norms, variate attention and the output stage. The
 * GEMMs, RMSNorm, SwiGLU and the epilogues are shared with other families in {@link Kernels}. Each
 * one is a device version of a {@code tirex2.cpu.*} stage and is tested against it.
 *
 * <p>All weights live in one packed {@link FloatArray} addressed by element offsets
 * ({@link PackedWeights}); activations are row-major {@code [rows · patches · width]}. Every kernel
 * is a {@link KernelContext} kernel launched with an explicit worker grid (no {@code @Parallel}
 * loops); element-wise ones map {@code ctx.globalIdx} to one item and bounds-check it.
 */
public final class TiRexKernels {

    /**
     * Scaler + patching ({@code cpu.InputStage}): one work-group of {@link Kernels#NORM_THREADS} threads
     * per row, with the sums reduced in local memory. The host uploads the values with NaN replaced
     * by 0 in {@code x} and the is-present flags in {@code present}: device compilers may assume
     * there are no NaNs (Metal's default fast math does), so a {@code v != v} test is not portable.
     * Writes the embedding input {@code tok [(r·patches + t)·2P + i]} (values, then the is-present
     * mask) and the scaler state {@code stats [r·3 + {0: loc, 1: scale, 2: binary}]}.
     */
    public static void scalePatch(KernelContext ctx, FloatArray x, FloatArray present, FloatArray tok, FloatArray stats,
            int rows, int len, int patch, int arcsinh, int binaryAware, float eps, HalfFloatArray th, int twin) {
        int r = ctx.groupIdx;
        int tid = ctx.localIdx;
        float[] sum = ctx.allocateFloatLocalArray(Kernels.NORM_THREADS);
        float[] cnt = ctx.allocateFloatLocalArray(Kernels.NORM_THREADS);
        float[] nonBinary = ctx.allocateFloatLocalArray(Kernels.NORM_THREADS);
        int base = r * len;
        float s0 = 0f, c0 = 0f, nb = 0f;
        for (int t = tid; t < len; t += Kernels.NORM_THREADS) {
            float v = x.get(base + t);
            float here = present.get(base + t);
            s0 += here * v;
            c0 += here;
            if (here != 0f && v != 0f && v != 1f) {
                nb = 1f;
            }
        }
        sum[tid] = s0;
        cnt[tid] = c0;
        nonBinary[tid] = nb;
        ctx.localBarrier();
        for (int st = Kernels.NORM_THREADS / 2; st > 0; st >>= 1) {
            if (tid < st) {
                sum[tid] += sum[tid + st];
                cnt[tid] += cnt[tid + st];
                nonBinary[tid] = TornadoMath.max(nonBinary[tid], nonBinary[tid + st]);
            }
            ctx.localBarrier();
        }
        float n = cnt[0];
        float loc = n == 0f ? 0f : sum[0] / n;
        int isBinary = binaryAware * (nonBinary[0] == 0f ? 1 : 0);
        ctx.localBarrier();
        float ss = 0f;
        for (int t = tid; t < len; t += Kernels.NORM_THREADS) {
            float d = x.get(base + t) - loc;
            ss += present.get(base + t) * d * d;
        }
        sum[tid] = ss;
        ctx.localBarrier();
        for (int st = Kernels.NORM_THREADS / 2; st > 0; st >>= 1) {
            if (tid < st) {
                sum[tid] += sum[tid + st];
            }
            ctx.localBarrier();
        }
        float scale = n == 0f ? 1f : TornadoMath.sqrt(sum[0] / n);
        scale = TornadoMath.max(scale, eps);
        if (isBinary == 1) {
            loc = 0f;
            scale = 1f;
        }
        if (tid == 0) {
            stats.set(r * 3, loc);
            stats.set(r * 3 + 1, scale);
            stats.set(r * 3 + 2, isBinary);
        }
        int doAsinh = arcsinh * (1 - isBinary);
        for (int pos = tid; pos < len; pos += Kernels.NORM_THREADS) {
            int t = pos / patch;
            int i = pos - t * patch;
            int o = (base / len * (len / patch) + t) * 2 * patch;
            float v = x.get(base + pos);
            float here = present.get(base + pos);
            float z = (v - loc) / scale;
            if (doAsinh == 1) {
                z = TornadoMath.asinh(z);
            }
            tok.set(o + i, here * z);
            tok.set(o + patch + i, here);
            if (twin == 1) {
                th.set(o + i, new HalfFloat(here * z));
                th.set(o + patch + i, new HalfFloat(here));
            }
        }
    }

    /**
     * Per-head LayerNorm (centred, biased variance, full-width weight), one work-group of
     * {@code dh} threads per (row, head). {@code mode} 0: {@code y = LN(x)}; 1: {@code y = σ(g) ·
     * LN(x)} with the gate read at {@code gOff + rowIndex·gStride}; 2: {@code y += LN(x)}.
     */
    public static void headLayerNorm(KernelContext ctx, FloatArray x, FloatArray y, FloatArray w, int wOff, int heads,
            int dh, float eps, FloatArray g, int gStride, int gOff, int mode, HalfFloatArray th, int twin) {
        int rh = ctx.groupIdx;
        int e = ctx.localIdx;
        int row = rh / heads;
        int h = rh - row * heads;
        float[] part = ctx.allocateFloatLocalArray(128);
        int d = heads * dh;
        int idx = row * d + h * dh + e;
        float v = x.get(idx);
        part[e] = v;
        ctx.localBarrier();
        for (int s = 64; s > 0; s >>= 1) {
            if (e < s) {
                part[e] += part[e + s];
            }
            ctx.localBarrier();
        }
        float mean = part[0] / dh;
        ctx.localBarrier();
        part[e] = (v - mean) * (v - mean);
        ctx.localBarrier();
        for (int s = 64; s > 0; s >>= 1) {
            if (e < s) {
                part[e] += part[e + s];
            }
            ctx.localBarrier();
        }
        float out = (v - mean) / TornadoMath.sqrt(part[0] / dh + eps) * w.get(wOff + h * dh + e);
        if (mode == 1) {
            float gate = g.get(gOff + row * gStride + h * dh + e);
            out = out / (1f + TornadoMath.exp(-gate));
        }
        if (mode == 2) {
            out += y.get(idx);
        }
        y.set(idx, out);
        if (twin == 1) {
            th.set(idx, new HalfFloat(out));
        }
    }

    /**
     * The mLSTM kernel keeps one step's scores in local memory: sequences up to this length (the
     * GIFT-Eval checkpoints use 285 patches, the base model 74).
     */
    public static final int MLSTM_MAX_SEQ = 512;

    /**
     * The mLSTM cell (parallel stabilised form, {@code cpu.Cells.mlstmCore}): one work-group of
     * {@code dv = 128} threads per (row, head, t). The ≤ t+1 scores and log-decays of step t are
     * built cooperatively in local memory, then thread j accumulates output column j. Reads q, k,
     * v and the i/f pre-activations from the fused projection {@code p [(r·seq + t)·stride + …]}
     * and writes {@code h [(r·seq + t)·heads·dv + head·dv + j]}.
     */
    public static void mlstmCell(KernelContext ctx, FloatArray p, FloatArray hOut, int rows, int seq, int heads,
            int stride, int kOff, int vOff, int iOff, int fOff, float cap, float eps) {
        int wg = ctx.groupIdx;
        int j = ctx.localIdx;
        int t = wg % seq;
        int rh = wg / seq;
        int h = rh % heads;
        int r = rh / heads;
        int dqk = 64;
        int dv = 128;
        float scale = 0.125f; // 1/sqrt(64)
        float[] cum = ctx.allocateFloatLocalArray(MLSTM_MAX_SEQ);
        float[] logD = ctx.allocateFloatLocalArray(MLSTM_MAX_SEQ);
        float[] score = ctx.allocateFloatLocalArray(MLSTM_MAX_SEQ);
        // logσ(capped f_s) and capped i_s for s ≤ t (128 threads stride over the steps)
        for (int s = j; s <= t; s += 128) {
            int so = (r * seq + s) * stride;
            float f = p.get(so + fOff + h);
            f = cap * TornadoMath.tanh(f / cap);
            cum[s] = TornadoMath.min(f, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(f)));
            float ig = p.get(so + iOff + h);
            logD[s] = cap * TornadoMath.tanh(ig / cap);
        }
        ctx.localBarrier();
        float[] fTotal = ctx.allocateFloatLocalArray(1);
        if (j == 0) {
            // logD[t, s] = F[t] − F[s] + i[s] with F[t] − F[s] = Σ_{s<u≤t} logσ(f_u) summed from t
            // backwards: a prefix sum F grows to hundreds over long sequences, and subtracting two of
            // them in FP32 costs ~1e-4 in the exponent exactly where the weights are largest
            float acc = 0f;
            for (int s = t; s >= 0; s--) {
                logD[s] = logD[s] + acc;
                acc += cum[s];
            }
            fTotal[0] = acc; // F[t], the zero initial state's stabiliser term
        }
        ctx.localBarrier();
        float ft = fTotal[0];
        int qo = (r * seq + t) * stride + h * dqk;
        for (int s = j; s <= t; s += 128) {
            int ko = (r * seq + s) * stride + kOff + h * dqk;
            float dot = 0f;
            for (int e = 0; e < dqk; e++) {
                dot += p.get(qo + e) * p.get(ko + e);
            }
            score[s] = dot * scale;
        }
        ctx.localBarrier();
        // m[t] = max(F[t], max_s logD[t, s]), computed redundantly by every thread
        float m = ft;
        for (int s = 0; s <= t; s++) {
            m = TornadoMath.max(m, logD[s]);
        }
        ctx.localBarrier();
        for (int s = j; s <= t; s += 128) {
            score[s] = score[s] * TornadoMath.exp(logD[s] - m);
        }
        ctx.localBarrier();
        float den = 0f;
        float num = 0f;
        int vo = r * seq * stride + vOff + h * dv + j;
        for (int s = 0; s <= t; s++) {
            float c = score[s];
            den += c;
            num += c * p.get(vo + s * stride);
        }
        float norm = TornadoMath.max(TornadoMath.abs(den), TornadoMath.exp(-m)) + eps;
        hOut.set((r * seq + t) * heads * dv + h * dv + j, num / norm);
    }

    /** Query steps per work-group of {@link #mlstmChunk}. */
    public static final int MLSTM_TB = 16;
    /** Key steps {@link #mlstmChunk} stages in local memory at a time. */
    public static final int MLSTM_SB = 16;

    /**
     * The mLSTM cell, chunkwise ({@link #mlstmCell} computes the same thing one step per
     * work-group). One work-group of {@code dv = 128} threads per (row, head, block of
     * {@link #MLSTM_TB} query steps). Keys and values stream through local memory in blocks of
     * {@link #MLSTM_SB}, and every staged key serves all the block's queries, so global traffic
     * falls by the query-block size: this is what matters at long contexts, where the per-step
     * kernel re-reads O(seq²) keys and values.
     *
     * <p>The stabiliser is kept online, as in flash attention: {@code m} starts at the zero
     * initial state's {@code F[t]} and grows with each key block, rescaling the running numerator
     * and denominator, so it ends at {@code max(F[t], max_s logD[t, s])} exactly as in the
     * per-step form. Log-decays are suffix sums relative to the block's last step {@code T}:
     * {@code logD[t, s] = A[s] − B[t]} with {@code A[s] = i_s + Σ_{s<u≤T} logσ(f_u)} and
     * {@code B[t] = Σ_{t<u≤T} logσ(f_u)} (at most {@link #MLSTM_TB} terms), which keeps the
     * cancellation small where the weights are large.
     */
    public static void mlstmChunk(KernelContext ctx, FloatArray p, FloatArray hOut, int rows, int seq, int heads,
            int stride, int kOff, int vOff, int iOff, int fOff, float cap, float eps) {
        int wg = ctx.groupIdx;
        int j = ctx.localIdx;
        int blocks = (seq + MLSTM_TB - 1) / MLSTM_TB;
        int tb = wg % blocks;
        int rh = wg / blocks;
        int h = rh % heads;
        int r = rh / heads;
        int dqk = 64;
        int dv = 128;
        float scale = 0.125f; // 1/sqrt(64)
        int t0 = tb * MLSTM_TB;
        int last = TornadoMath.min(t0 + MLSTM_TB, seq) - 1; // T, the block's last step
        float[] a = ctx.allocateFloatLocalArray(MLSTM_MAX_SEQ);  // logσ(f_s), then A[s]
        float[] ig = ctx.allocateFloatLocalArray(MLSTM_MAX_SEQ); // capped i_s
        float[] q = ctx.allocateFloatLocalArray(MLSTM_TB * 64);
        float[] k = ctx.allocateFloatLocalArray(MLSTM_SB * 64);
        float[] v = ctx.allocateFloatLocalArray(MLSTM_SB * 128);
        float[] w = ctx.allocateFloatLocalArray(MLSTM_TB * MLSTM_SB); // scores, then weights
        float[] lg = ctx.allocateFloatLocalArray(MLSTM_TB * MLSTM_SB);
        float[] num = ctx.allocateFloatLocalArray(MLSTM_TB * 128);
        float[] bq = ctx.allocateFloatLocalArray(MLSTM_TB);  // B[t]
        float[] mq = ctx.allocateFloatLocalArray(MLSTM_TB);  // running stabiliser
        float[] den = ctx.allocateFloatLocalArray(MLSTM_TB);
        float[] fac = ctx.allocateFloatLocalArray(MLSTM_TB); // this block's rescale factor
        float[] ftot = ctx.allocateFloatLocalArray(1);

        for (int s = j; s <= last; s += 128) {
            int so = (r * seq + s) * stride;
            float f = p.get(so + fOff + h);
            f = cap * TornadoMath.tanh(f / cap);
            a[s] = TornadoMath.min(f, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(f)));
            float gi = p.get(so + iOff + h);
            ig[s] = cap * TornadoMath.tanh(gi / cap);
        }
        for (int e = j; e < MLSTM_TB * dqk; e += 128) {
            int tq = e / dqk;
            float qv = 0f;
            if (t0 + tq < seq) {
                qv = p.get((r * seq + t0 + tq) * stride + h * dqk + e % dqk);
            }
            q[e] = qv;
        }
        for (int e = j; e < MLSTM_TB * dv; e += 128) {
            num[e] = 0f;
        }
        if (j < MLSTM_TB) {
            bq[j] = 0f;
        }
        ctx.localBarrier();
        if (j == 0) {
            // suffix sums from T backwards: A[s] = i_s + Σ_{s<u≤T} logσ(f_u), B[t], Σ_{u≤T} logσ(f_u)
            float acc = 0f;
            for (int s = last; s >= 0; s--) {
                float lf = a[s];
                if (s >= t0) {
                    bq[s - t0] = acc;
                }
                a[s] = ig[s] + acc;
                acc += lf;
            }
            ftot[0] = acc;
        }
        ctx.localBarrier();
        if (j < MLSTM_TB) {
            mq[j] = ftot[0] - bq[j]; // F[t], the zero initial state's stabiliser term
            den[j] = 0f;
        }
        ctx.localBarrier();

        for (int s0 = 0; s0 <= last; s0 += MLSTM_SB) {
            for (int e = j; e < MLSTM_SB * dqk; e += 128) {
                int sk = s0 + e / dqk;
                float kv = 0f;
                if (sk <= last) {
                    kv = p.get((r * seq + sk) * stride + kOff + h * dqk + e % dqk);
                }
                k[e] = kv;
            }
            for (int e = j; e < MLSTM_SB * dv; e += 128) {
                int sk = s0 + e / dv;
                float vv = 0f;
                if (sk <= last) {
                    vv = p.get((r * seq + sk) * stride + vOff + h * dv + e % dv);
                }
                v[e] = vv;
            }
            ctx.localBarrier();
            // scores and log-decays of the block's (query, key) pairs, two per thread
            for (int pr = j; pr < MLSTM_TB * MLSTM_SB; pr += 128) {
                int tq = pr / MLSTM_SB;
                int sk = pr % MLSTM_SB;
                float dot = 0f;
                for (int e = 0; e < dqk; e++) {
                    dot += q[tq * dqk + e] * k[sk * dqk + e];
                }
                w[pr] = dot * scale;
                float l = -1e30f;
                int s = s0 + sk;
                if (s <= t0 + tq) {
                    if (s <= last) {
                        l = a[s] - bq[tq];
                    }
                }
                lg[pr] = l;
            }
            ctx.localBarrier();
            // online stabiliser: one thread per query
            if (j < MLSTM_TB) {
                float mOld = mq[j];
                float mNew = mOld;
                for (int sk = 0; sk < MLSTM_SB; sk++) {
                    mNew = TornadoMath.max(mNew, lg[j * MLSTM_SB + sk]);
                }
                float f = TornadoMath.exp(mOld - mNew);
                float d = den[j] * f;
                for (int sk = 0; sk < MLSTM_SB; sk++) {
                    float l = lg[j * MLSTM_SB + sk];
                    float c = 0f;
                    if (l > -1e29f) {
                        c = w[j * MLSTM_SB + sk] * TornadoMath.exp(l - mNew);
                    }
                    w[j * MLSTM_SB + sk] = c;
                    d += c;
                }
                den[j] = d;
                mq[j] = mNew;
                fac[j] = f;
            }
            ctx.localBarrier();
            // thread j accumulates output column j for every query of the block
            for (int tq = 0; tq < MLSTM_TB; tq++) {
                float acc = num[tq * dv + j] * fac[tq];
                for (int sk = 0; sk < MLSTM_SB; sk++) {
                    acc += w[tq * MLSTM_SB + sk] * v[sk * dv + j];
                }
                num[tq * dv + j] = acc;
            }
            ctx.localBarrier();
        }
        for (int tq = 0; tq < MLSTM_TB; tq++) {
            int t = t0 + tq;
            if (t < seq) {
                float norm = TornadoMath.max(TornadoMath.abs(den[tq]), TornadoMath.exp(-mq[tq])) + eps;
                hOut.set((r * seq + t) * heads * dv + h * dv + j, num[tq * dv + j] / norm);
            }
        }
    }

    /** Rows per work-group the sLSTM recurrence can hold in local memory. */
    public static final int SLSTM_MAX_ROWS = 8;

    /**
     * The sLSTM recurrence ({@code cpu.Cells.slstmRecurrence}). One work-group per (block of
     * {@code rowsPerGroup} rows, head) runs all {@code seq} steps; heads are independent because R
     * is block-diagonal, so no work-group waits on another. Its {@code 4·dh / gatesPerThread}
     * threads split the four gates: thread {@code (gBlock, e)} computes gates {@code gBlock·gpt …}
     * of lane e for every row of the block, so each R element read (coalesced over e) serves
     * {@code rowsPerGroup} rows. The raw gates go to local memory; threads {@code e < dh} then
     * update the stabilised state of each row and publish y for the next step.
     */
    public static void slstmRecurrence(KernelContext ctx, FloatArray wx, FloatArray w, int rOff, int bOff, FloatArray y,
            int rows, int seq, int heads, int gatesPerThread, int rowsPerGroup) {
        int wg = ctx.groupIdx;
        int tid = ctx.localIdx;
        int h = wg % heads;
        int row0 = (wg / heads) * rowsPerGroup;
        int dh = 128;
        int d = heads * dh;
        int e = tid % dh;
        int g0 = (tid / dh) * gatesPerThread;
        float[] yl = ctx.allocateFloatLocalArray(SLSTM_MAX_ROWS * 128);
        float[] raw = ctx.allocateFloatLocalArray(SLSTM_MAX_ROWS * 512);
        float[] c = new float[SLSTM_MAX_ROWS];
        float[] nn = new float[SLSTM_MAX_ROWS];
        float[] m = new float[SLSTM_MAX_ROWS];
        float[] acc = new float[SLSTM_MAX_ROWS];
        for (int rb = 0; rb < SLSTM_MAX_ROWS; rb++) {
            c[rb] = 0f;
            nn[rb] = 0f;
            m[rb] = 0f;
        }
        if (tid < dh) {
            for (int rb = 0; rb < rowsPerGroup; rb++) {
                yl[rb * dh + tid] = 0f;
            }
        }
        ctx.localBarrier();
        for (int t = 0; t < seq; t++) {
            for (int gi = 0; gi < gatesPerThread; gi++) {
                int g = g0 + gi;
                float bias = w.get(bOff + (h * 4 + g) * dh + e);
                for (int rb = 0; rb < rowsPerGroup; rb++) {
                    int row = TornadoMath.min(row0 + rb, rows - 1);
                    acc[rb] = wx.get((row * seq + t) * 4 * d + g * d + h * dh + e) + bias;
                }
                int ro = rOff + (h * dh * 4 + g) * dh + e;
                for (int pp = 0; pp < dh; pp++) {
                    float r = w.get(ro + pp * 4 * dh);
                    for (int rb = 0; rb < rowsPerGroup; rb++) {
                        acc[rb] += yl[rb * dh + pp] * r;
                    }
                }
                for (int rb = 0; rb < rowsPerGroup; rb++) {
                    raw[rb * 512 + g * dh + e] = acc[rb];
                }
            }
            ctx.localBarrier();
            if (tid < dh) {
                for (int rb = 0; rb < rowsPerGroup; rb++) {
                    int o = rb * 512;
                    float ri = raw[o + e];
                    float rf = raw[o + dh + e];
                    float rz = raw[o + 2 * dh + e];
                    float rog = raw[o + 3 * dh + e];
                    float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m[rb];
                    float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
                    float ig = TornadoMath.exp(ri - mNew);
                    float fg = TornadoMath.exp(logf - mNew);
                    c[rb] = fg * c[rb] + ig * TornadoMath.tanh(rz);
                    nn[rb] = TornadoMath.max(fg * nn[rb] + ig, 1f);
                    float yNew = c[rb] / nn[rb] / (1f + TornadoMath.exp(-rog));
                    m[rb] = mNew;
                    yl[rb * dh + e] = yNew;
                    int row = row0 + rb;
                    if (row < rows) {
                        y.set((row * seq + t) * d + h * dh + e, yNew);
                    }
                }
            }
            ctx.localBarrier();
        }
    }

    /**
     * {@link #slstmRecurrence} with R read from an FP16 copy {@code rh} (offset {@code rOff}),
     * halving the bytes each step streams; the state and accumulation stay FP32.
     */
    public static void slstmRecurrenceHalfR(KernelContext ctx, FloatArray wx, FloatArray w, HalfFloatArray rh, int rOff,
            int bOff, FloatArray y,
            int rows, int seq, int heads, int gatesPerThread, int rowsPerGroup) {
        int wg = ctx.groupIdx;
        int tid = ctx.localIdx;
        int h = wg % heads;
        int row0 = (wg / heads) * rowsPerGroup;
        int dh = 128;
        int d = heads * dh;
        int e = tid % dh;
        int g0 = (tid / dh) * gatesPerThread;
        float[] yl = ctx.allocateFloatLocalArray(SLSTM_MAX_ROWS * 128);
        float[] raw = ctx.allocateFloatLocalArray(SLSTM_MAX_ROWS * 512);
        float[] c = new float[SLSTM_MAX_ROWS];
        float[] nn = new float[SLSTM_MAX_ROWS];
        float[] m = new float[SLSTM_MAX_ROWS];
        float[] acc = new float[SLSTM_MAX_ROWS];
        for (int rb = 0; rb < SLSTM_MAX_ROWS; rb++) {
            c[rb] = 0f;
            nn[rb] = 0f;
            m[rb] = 0f;
        }
        if (tid < dh) {
            for (int rb = 0; rb < rowsPerGroup; rb++) {
                yl[rb * dh + tid] = 0f;
            }
        }
        ctx.localBarrier();
        for (int t = 0; t < seq; t++) {
            for (int gi = 0; gi < gatesPerThread; gi++) {
                int g = g0 + gi;
                float bias = w.get(bOff + (h * 4 + g) * dh + e);
                for (int rb = 0; rb < rowsPerGroup; rb++) {
                    int row = TornadoMath.min(row0 + rb, rows - 1);
                    acc[rb] = wx.get((row * seq + t) * 4 * d + g * d + h * dh + e) + bias;
                }
                int ro = rOff + (h * dh * 4 + g) * dh + e;
                for (int pp = 0; pp < dh; pp++) {
                    float r = rh.get(ro + pp * 4 * dh).getFloat32();
                    for (int rb = 0; rb < rowsPerGroup; rb++) {
                        acc[rb] += yl[rb * dh + pp] * r;
                    }
                }
                for (int rb = 0; rb < rowsPerGroup; rb++) {
                    raw[rb * 512 + g * dh + e] = acc[rb];
                }
            }
            ctx.localBarrier();
            if (tid < dh) {
                for (int rb = 0; rb < rowsPerGroup; rb++) {
                    int o = rb * 512;
                    float ri = raw[o + e];
                    float rf = raw[o + dh + e];
                    float rz = raw[o + 2 * dh + e];
                    float rog = raw[o + 3 * dh + e];
                    float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m[rb];
                    float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
                    float ig = TornadoMath.exp(ri - mNew);
                    float fg = TornadoMath.exp(logf - mNew);
                    c[rb] = fg * c[rb] + ig * TornadoMath.tanh(rz);
                    nn[rb] = TornadoMath.max(fg * nn[rb] + ig, 1f);
                    float yNew = c[rb] / nn[rb] / (1f + TornadoMath.exp(-rog));
                    m[rb] = mNew;
                    yl[rb * dh + e] = yNew;
                    int row = row0 + rb;
                    if (row < rows) {
                        y.set((row * seq + t) * d + h * dh + e, yNew);
                    }
                }
            }
            ctx.localBarrier();
        }
    }

    /** R values per thread that {@link #slstmRecurrenceRegR} keeps in registers (the rest are streamed). */
    public static final int SLSTM_REG_R = 64;

    /**
     * {@link #slstmRecurrenceHalfR} with the first {@link #SLSTM_REG_R} of each thread's 128 R values
     * held in registers for the whole sequence, so each step streams only the other half of R
     * (the step is bound by streaming R from L2: 128 KB per head per step in FP16). One thread per
     * (gate, lane): exactly {@code 4·dh = 512} threads per work-group (gatesPerThread = 1), so
     * it needs a device that allows 512-thread work-groups. The registers are named scalars
     * rather than a private array, which GPU compilers would move to local memory when indexed
     * in a loop. Generated: the loads and the unrolled dot products follow one pattern.
     */
    public static void slstmRecurrenceRegR(KernelContext ctx, FloatArray wx, FloatArray w, HalfFloatArray rh, int rOff,
            int bOff, FloatArray y, int rows, int seq, int heads, int rowsPerGroup) {
        int wg = ctx.groupIdx;
        int tid = ctx.localIdx;
        int h = wg % heads;
        int row0 = (wg / heads) * rowsPerGroup;
        int dh = 128;
        int d = heads * dh;
        int e = tid % dh;
        int g = tid / dh;
        float[] yl = ctx.allocateFloatLocalArray(SLSTM_MAX_ROWS * 128);
        float[] raw = ctx.allocateFloatLocalArray(SLSTM_MAX_ROWS * 512);
        // the per-row cell state is scalars, one per row of SLSTM_MAX_ROWS = 8, not arrays: kernel
        // compilers that do not unroll the row loops would otherwise keep them in memory
        float c0 = 0f, c1 = 0f, c2 = 0f, c3 = 0f, c4 = 0f, c5 = 0f, c6 = 0f, c7 = 0f;
        float n0 = 0f, n1 = 0f, n2 = 0f, n3 = 0f, n4 = 0f, n5 = 0f, n6 = 0f, n7 = 0f;
        float m0 = 0f, m1 = 0f, m2 = 0f, m3 = 0f, m4 = 0f, m5 = 0f, m6 = 0f, m7 = 0f;
        if (tid < dh) {
            for (int rb = 0; rb < rowsPerGroup; rb++) {
                yl[rb * dh + tid] = 0f;
            }
        }
        int ro = rOff + (h * dh * 4 + g) * dh + e;
        float bias = w.get(bOff + (h * 4 + g) * dh + e);
        float r0 = rh.get(ro + 0 * 4 * dh).getFloat32();
        float r1 = rh.get(ro + 1 * 4 * dh).getFloat32();
        float r2 = rh.get(ro + 2 * 4 * dh).getFloat32();
        float r3 = rh.get(ro + 3 * 4 * dh).getFloat32();
        float r4 = rh.get(ro + 4 * 4 * dh).getFloat32();
        float r5 = rh.get(ro + 5 * 4 * dh).getFloat32();
        float r6 = rh.get(ro + 6 * 4 * dh).getFloat32();
        float r7 = rh.get(ro + 7 * 4 * dh).getFloat32();
        float r8 = rh.get(ro + 8 * 4 * dh).getFloat32();
        float r9 = rh.get(ro + 9 * 4 * dh).getFloat32();
        float r10 = rh.get(ro + 10 * 4 * dh).getFloat32();
        float r11 = rh.get(ro + 11 * 4 * dh).getFloat32();
        float r12 = rh.get(ro + 12 * 4 * dh).getFloat32();
        float r13 = rh.get(ro + 13 * 4 * dh).getFloat32();
        float r14 = rh.get(ro + 14 * 4 * dh).getFloat32();
        float r15 = rh.get(ro + 15 * 4 * dh).getFloat32();
        float r16 = rh.get(ro + 16 * 4 * dh).getFloat32();
        float r17 = rh.get(ro + 17 * 4 * dh).getFloat32();
        float r18 = rh.get(ro + 18 * 4 * dh).getFloat32();
        float r19 = rh.get(ro + 19 * 4 * dh).getFloat32();
        float r20 = rh.get(ro + 20 * 4 * dh).getFloat32();
        float r21 = rh.get(ro + 21 * 4 * dh).getFloat32();
        float r22 = rh.get(ro + 22 * 4 * dh).getFloat32();
        float r23 = rh.get(ro + 23 * 4 * dh).getFloat32();
        float r24 = rh.get(ro + 24 * 4 * dh).getFloat32();
        float r25 = rh.get(ro + 25 * 4 * dh).getFloat32();
        float r26 = rh.get(ro + 26 * 4 * dh).getFloat32();
        float r27 = rh.get(ro + 27 * 4 * dh).getFloat32();
        float r28 = rh.get(ro + 28 * 4 * dh).getFloat32();
        float r29 = rh.get(ro + 29 * 4 * dh).getFloat32();
        float r30 = rh.get(ro + 30 * 4 * dh).getFloat32();
        float r31 = rh.get(ro + 31 * 4 * dh).getFloat32();
        float r32 = rh.get(ro + 32 * 4 * dh).getFloat32();
        float r33 = rh.get(ro + 33 * 4 * dh).getFloat32();
        float r34 = rh.get(ro + 34 * 4 * dh).getFloat32();
        float r35 = rh.get(ro + 35 * 4 * dh).getFloat32();
        float r36 = rh.get(ro + 36 * 4 * dh).getFloat32();
        float r37 = rh.get(ro + 37 * 4 * dh).getFloat32();
        float r38 = rh.get(ro + 38 * 4 * dh).getFloat32();
        float r39 = rh.get(ro + 39 * 4 * dh).getFloat32();
        float r40 = rh.get(ro + 40 * 4 * dh).getFloat32();
        float r41 = rh.get(ro + 41 * 4 * dh).getFloat32();
        float r42 = rh.get(ro + 42 * 4 * dh).getFloat32();
        float r43 = rh.get(ro + 43 * 4 * dh).getFloat32();
        float r44 = rh.get(ro + 44 * 4 * dh).getFloat32();
        float r45 = rh.get(ro + 45 * 4 * dh).getFloat32();
        float r46 = rh.get(ro + 46 * 4 * dh).getFloat32();
        float r47 = rh.get(ro + 47 * 4 * dh).getFloat32();
        float r48 = rh.get(ro + 48 * 4 * dh).getFloat32();
        float r49 = rh.get(ro + 49 * 4 * dh).getFloat32();
        float r50 = rh.get(ro + 50 * 4 * dh).getFloat32();
        float r51 = rh.get(ro + 51 * 4 * dh).getFloat32();
        float r52 = rh.get(ro + 52 * 4 * dh).getFloat32();
        float r53 = rh.get(ro + 53 * 4 * dh).getFloat32();
        float r54 = rh.get(ro + 54 * 4 * dh).getFloat32();
        float r55 = rh.get(ro + 55 * 4 * dh).getFloat32();
        float r56 = rh.get(ro + 56 * 4 * dh).getFloat32();
        float r57 = rh.get(ro + 57 * 4 * dh).getFloat32();
        float r58 = rh.get(ro + 58 * 4 * dh).getFloat32();
        float r59 = rh.get(ro + 59 * 4 * dh).getFloat32();
        float r60 = rh.get(ro + 60 * 4 * dh).getFloat32();
        float r61 = rh.get(ro + 61 * 4 * dh).getFloat32();
        float r62 = rh.get(ro + 62 * 4 * dh).getFloat32();
        float r63 = rh.get(ro + 63 * 4 * dh).getFloat32();
        ctx.localBarrier();
        for (int t = 0; t < seq; t++) {
            for (int rb = 0; rb < rowsPerGroup; rb++) {
                int row = TornadoMath.min(row0 + rb, rows - 1);
                int yo = rb * dh;
                float a = wx.get((row * seq + t) * 4 * d + g * d + h * dh + e) + bias;
                a += yl[yo + 0] * r0;
                a += yl[yo + 1] * r1;
                a += yl[yo + 2] * r2;
                a += yl[yo + 3] * r3;
                a += yl[yo + 4] * r4;
                a += yl[yo + 5] * r5;
                a += yl[yo + 6] * r6;
                a += yl[yo + 7] * r7;
                a += yl[yo + 8] * r8;
                a += yl[yo + 9] * r9;
                a += yl[yo + 10] * r10;
                a += yl[yo + 11] * r11;
                a += yl[yo + 12] * r12;
                a += yl[yo + 13] * r13;
                a += yl[yo + 14] * r14;
                a += yl[yo + 15] * r15;
                a += yl[yo + 16] * r16;
                a += yl[yo + 17] * r17;
                a += yl[yo + 18] * r18;
                a += yl[yo + 19] * r19;
                a += yl[yo + 20] * r20;
                a += yl[yo + 21] * r21;
                a += yl[yo + 22] * r22;
                a += yl[yo + 23] * r23;
                a += yl[yo + 24] * r24;
                a += yl[yo + 25] * r25;
                a += yl[yo + 26] * r26;
                a += yl[yo + 27] * r27;
                a += yl[yo + 28] * r28;
                a += yl[yo + 29] * r29;
                a += yl[yo + 30] * r30;
                a += yl[yo + 31] * r31;
                a += yl[yo + 32] * r32;
                a += yl[yo + 33] * r33;
                a += yl[yo + 34] * r34;
                a += yl[yo + 35] * r35;
                a += yl[yo + 36] * r36;
                a += yl[yo + 37] * r37;
                a += yl[yo + 38] * r38;
                a += yl[yo + 39] * r39;
                a += yl[yo + 40] * r40;
                a += yl[yo + 41] * r41;
                a += yl[yo + 42] * r42;
                a += yl[yo + 43] * r43;
                a += yl[yo + 44] * r44;
                a += yl[yo + 45] * r45;
                a += yl[yo + 46] * r46;
                a += yl[yo + 47] * r47;
                a += yl[yo + 48] * r48;
                a += yl[yo + 49] * r49;
                a += yl[yo + 50] * r50;
                a += yl[yo + 51] * r51;
                a += yl[yo + 52] * r52;
                a += yl[yo + 53] * r53;
                a += yl[yo + 54] * r54;
                a += yl[yo + 55] * r55;
                a += yl[yo + 56] * r56;
                a += yl[yo + 57] * r57;
                a += yl[yo + 58] * r58;
                a += yl[yo + 59] * r59;
                a += yl[yo + 60] * r60;
                a += yl[yo + 61] * r61;
                a += yl[yo + 62] * r62;
                a += yl[yo + 63] * r63;
                raw[rb * 512 + g * dh + e] = a;
            }
            float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f, s4 = 0f, s5 = 0f, s6 = 0f, s7 = 0f;
            for (int pp = SLSTM_REG_R; pp < dh; pp++) {
                float r = rh.get(ro + pp * 4 * dh).getFloat32();
                s0 += yl[pp] * r;
                if (1 < rowsPerGroup) {
                    s1 += yl[128 + pp] * r;
                }
                if (2 < rowsPerGroup) {
                    s2 += yl[256 + pp] * r;
                }
                if (3 < rowsPerGroup) {
                    s3 += yl[384 + pp] * r;
                }
                if (4 < rowsPerGroup) {
                    s4 += yl[512 + pp] * r;
                }
                if (5 < rowsPerGroup) {
                    s5 += yl[640 + pp] * r;
                }
                if (6 < rowsPerGroup) {
                    s6 += yl[768 + pp] * r;
                }
                if (7 < rowsPerGroup) {
                    s7 += yl[896 + pp] * r;
                }
            }
            raw[g * dh + e] += s0;
            if (1 < rowsPerGroup) {
                raw[512 + g * dh + e] += s1;
            }
            if (2 < rowsPerGroup) {
                raw[1024 + g * dh + e] += s2;
            }
            if (3 < rowsPerGroup) {
                raw[1536 + g * dh + e] += s3;
            }
            if (4 < rowsPerGroup) {
                raw[2048 + g * dh + e] += s4;
            }
            if (5 < rowsPerGroup) {
                raw[2560 + g * dh + e] += s5;
            }
            if (6 < rowsPerGroup) {
                raw[3072 + g * dh + e] += s6;
            }
            if (7 < rowsPerGroup) {
                raw[3584 + g * dh + e] += s7;
            }
            ctx.localBarrier();
            if (tid < dh) {
                if (0 < rowsPerGroup) {
                    float ri = raw[0 + e];
                    float rf = raw[0 + dh + e];
                    float rz = raw[0 + 2 * dh + e];
                    float rog = raw[0 + 3 * dh + e];
                    float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m0;
                    float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
                    float ig = TornadoMath.exp(ri - mNew);
                    float fg = TornadoMath.exp(logf - mNew);
                    c0 = fg * c0 + ig * TornadoMath.tanh(rz);
                    n0 = TornadoMath.max(fg * n0 + ig, 1f);
                    float yNew = c0 / n0 / (1f + TornadoMath.exp(-rog));
                    m0 = mNew;
                    yl[0 + e] = yNew;
                    if (row0 + 0 < rows) {
                        y.set(((row0 + 0) * seq + t) * d + h * dh + e, yNew);
                    }
                }
                if (1 < rowsPerGroup) {
                    float ri = raw[512 + e];
                    float rf = raw[512 + dh + e];
                    float rz = raw[512 + 2 * dh + e];
                    float rog = raw[512 + 3 * dh + e];
                    float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m1;
                    float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
                    float ig = TornadoMath.exp(ri - mNew);
                    float fg = TornadoMath.exp(logf - mNew);
                    c1 = fg * c1 + ig * TornadoMath.tanh(rz);
                    n1 = TornadoMath.max(fg * n1 + ig, 1f);
                    float yNew = c1 / n1 / (1f + TornadoMath.exp(-rog));
                    m1 = mNew;
                    yl[128 + e] = yNew;
                    if (row0 + 1 < rows) {
                        y.set(((row0 + 1) * seq + t) * d + h * dh + e, yNew);
                    }
                }
                if (2 < rowsPerGroup) {
                    float ri = raw[1024 + e];
                    float rf = raw[1024 + dh + e];
                    float rz = raw[1024 + 2 * dh + e];
                    float rog = raw[1024 + 3 * dh + e];
                    float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m2;
                    float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
                    float ig = TornadoMath.exp(ri - mNew);
                    float fg = TornadoMath.exp(logf - mNew);
                    c2 = fg * c2 + ig * TornadoMath.tanh(rz);
                    n2 = TornadoMath.max(fg * n2 + ig, 1f);
                    float yNew = c2 / n2 / (1f + TornadoMath.exp(-rog));
                    m2 = mNew;
                    yl[256 + e] = yNew;
                    if (row0 + 2 < rows) {
                        y.set(((row0 + 2) * seq + t) * d + h * dh + e, yNew);
                    }
                }
                if (3 < rowsPerGroup) {
                    float ri = raw[1536 + e];
                    float rf = raw[1536 + dh + e];
                    float rz = raw[1536 + 2 * dh + e];
                    float rog = raw[1536 + 3 * dh + e];
                    float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m3;
                    float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
                    float ig = TornadoMath.exp(ri - mNew);
                    float fg = TornadoMath.exp(logf - mNew);
                    c3 = fg * c3 + ig * TornadoMath.tanh(rz);
                    n3 = TornadoMath.max(fg * n3 + ig, 1f);
                    float yNew = c3 / n3 / (1f + TornadoMath.exp(-rog));
                    m3 = mNew;
                    yl[384 + e] = yNew;
                    if (row0 + 3 < rows) {
                        y.set(((row0 + 3) * seq + t) * d + h * dh + e, yNew);
                    }
                }
                if (4 < rowsPerGroup) {
                    float ri = raw[2048 + e];
                    float rf = raw[2048 + dh + e];
                    float rz = raw[2048 + 2 * dh + e];
                    float rog = raw[2048 + 3 * dh + e];
                    float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m4;
                    float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
                    float ig = TornadoMath.exp(ri - mNew);
                    float fg = TornadoMath.exp(logf - mNew);
                    c4 = fg * c4 + ig * TornadoMath.tanh(rz);
                    n4 = TornadoMath.max(fg * n4 + ig, 1f);
                    float yNew = c4 / n4 / (1f + TornadoMath.exp(-rog));
                    m4 = mNew;
                    yl[512 + e] = yNew;
                    if (row0 + 4 < rows) {
                        y.set(((row0 + 4) * seq + t) * d + h * dh + e, yNew);
                    }
                }
                if (5 < rowsPerGroup) {
                    float ri = raw[2560 + e];
                    float rf = raw[2560 + dh + e];
                    float rz = raw[2560 + 2 * dh + e];
                    float rog = raw[2560 + 3 * dh + e];
                    float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m5;
                    float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
                    float ig = TornadoMath.exp(ri - mNew);
                    float fg = TornadoMath.exp(logf - mNew);
                    c5 = fg * c5 + ig * TornadoMath.tanh(rz);
                    n5 = TornadoMath.max(fg * n5 + ig, 1f);
                    float yNew = c5 / n5 / (1f + TornadoMath.exp(-rog));
                    m5 = mNew;
                    yl[640 + e] = yNew;
                    if (row0 + 5 < rows) {
                        y.set(((row0 + 5) * seq + t) * d + h * dh + e, yNew);
                    }
                }
                if (6 < rowsPerGroup) {
                    float ri = raw[3072 + e];
                    float rf = raw[3072 + dh + e];
                    float rz = raw[3072 + 2 * dh + e];
                    float rog = raw[3072 + 3 * dh + e];
                    float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m6;
                    float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
                    float ig = TornadoMath.exp(ri - mNew);
                    float fg = TornadoMath.exp(logf - mNew);
                    c6 = fg * c6 + ig * TornadoMath.tanh(rz);
                    n6 = TornadoMath.max(fg * n6 + ig, 1f);
                    float yNew = c6 / n6 / (1f + TornadoMath.exp(-rog));
                    m6 = mNew;
                    yl[768 + e] = yNew;
                    if (row0 + 6 < rows) {
                        y.set(((row0 + 6) * seq + t) * d + h * dh + e, yNew);
                    }
                }
                if (7 < rowsPerGroup) {
                    float ri = raw[3584 + e];
                    float rf = raw[3584 + dh + e];
                    float rz = raw[3584 + 2 * dh + e];
                    float rog = raw[3584 + 3 * dh + e];
                    float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m7;
                    float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
                    float ig = TornadoMath.exp(ri - mNew);
                    float fg = TornadoMath.exp(logf - mNew);
                    c7 = fg * c7 + ig * TornadoMath.tanh(rz);
                    n7 = TornadoMath.max(fg * n7 + ig, 1f);
                    float yNew = c7 / n7 / (1f + TornadoMath.exp(-rog));
                    m7 = mNew;
                    yl[896 + e] = yNew;
                    if (row0 + 7 < rows) {
                        y.set(((row0 + 7) * seq + t) * d + h * dh + e, yNew);
                    }
                }
            }
            ctx.localBarrier();
        }
    }

    /**
     * {@code y [(r·patches + t)·Q·P + q·P + i]} → {@code out [r·Q·len + q·len + t·P + i]}, then
     * {@code sinh(clamp(y, ±clamp))·scale + loc} (no sinh on binary rows or without arcsinh).
     */
    public static void unpatch(KernelContext ctx, FloatArray y, FloatArray stats, FloatArray out, int rows, int patches, int nq,
            int patch, int arcsinh, float clamp) {
        int len = patches * patch;
        int id = ctx.globalIdx;
        if (id < rows * nq * len) {
            int r = id / (nq * len);
            int rem = id - r * nq * len;
            int q = rem / len;
            int pos = rem - q * len;
            int t = pos / patch;
            int i = pos - t * patch;
            float v = y.get((r * patches + t) * nq * patch + q * patch + i);
            if (arcsinh == 1 && stats.get(r * 3 + 2) == 0f) {
                float cv = TornadoMath.clamp(v, -clamp, clamp);
                v = (TornadoMath.exp(cv) - TornadoMath.exp(-cv)) * 0.5f;
            }
            out.set(id, v * stats.get(r * 3 + 1) + stats.get(r * 3));
        }
    }
    // ---- P2: covariates and multivariate --------------------------------------------------------

    /**
     * {@code dst[(k·seq + t)·d + j] = src[(idx[k]·seq + t')·d + j]} with {@code t' = seq−1−t} when
     * {@code flip} is 1 (the reverse cell pass reads the known covariates back to front).
     */
    public static void gatherRows(KernelContext ctx, FloatArray src, FloatArray dst, IntArray idx, int count, int seq, int d, int flip, HalfFloatArray th, int twin) {
        int id = ctx.globalIdx;
        if (id < count * seq * d) {
            int k = id / (seq * d);
            int rem = id - k * seq * d;
            int t = rem / d;
            int j = rem - t * d;
            int ts = flip == 1 ? seq - 1 - t : t;
            float v = src.get((idx.get(k) * seq + ts) * d + j);
            dst.set(id, v);
            if (twin == 1) {
                th.set(id, new HalfFloat(v));
            }
        }
    }

    /** {@code dst[(idx[k]·seq + t)·d + j] = src[(k·seq + t)·d + j]}. */
    public static void scatterRows(KernelContext ctx, FloatArray src, FloatArray dst, IntArray idx, int count, int seq, int d) {
        int id = ctx.globalIdx;
        if (id < count * seq * d) {
            int k = id / (seq * d);
            int rem = id - k * seq * d;
            int t = rem / d;
            int j = rem - t * d;
            dst.set((idx.get(k) * seq + t) * d + j, src.get(id));
        }
    }

    /**
     * The combination input of the known covariates: {@code cat[(k·seq + t)·2d] = y[known[k], t] |
     * rev[k, seq−1−t]} (the reverse output is flipped back to forward time).
     */
    public static void concatForward(KernelContext ctx, FloatArray y, FloatArray rev, IntArray known, FloatArray cat, int count, int seq, int d, HalfFloatArray th, int twin) {
        int id = ctx.globalIdx;
        if (id < count * seq * 2 * d) {
            int k = id / (seq * 2 * d);
            int rem = id - k * seq * 2 * d;
            int t = rem / (2 * d);
            int j = rem - t * 2 * d;
            float v;
            if (j < d) {
                v = y.get((known.get(k) * seq + t) * d + j);
            } else {
                v = rev.get((k * seq + (seq - 1 - t)) * d + j - d);
            }
            cat.set(id, v);
            if (twin == 1) {
                th.set(id, new HalfFloat(v));
            }
        }
    }

    /** Local memory of {@link #qkNorm}: up to 8 vectors of 128 reduced concurrently. */
    public static final int QKNORM_MAX_THREADS = 1024;

    /**
     * RMS-normalises q and k of the fused {@code q | k | v} projection in place, per head, with no
     * weight ({@code F.rms_norm}). One work-group per token; its {@code vectorsPerPass·128} threads
     * reduce that many of the {@code 2·heads} vectors at once (one 128-thread segment each, element
     * j per thread, coalesced), so on a device allowing 1024 threads a token takes one pass.
     */
    public static void qkNorm(KernelContext ctx, FloatArray p, int tokens, int heads, int dh, int stride, float eps,
            int vectorsPerPass) {
        int tok = ctx.groupIdx;
        int tid = ctx.localIdx;
        int seg = tid / 128;
        int j = tid - seg * 128;
        float[] part = ctx.allocateFloatLocalArray(QKNORM_MAX_THREADS);
        for (int base = 0; base < 2 * heads; base += vectorsPerPass) {
            int vec = base + seg; // vec = which·heads + h, and q | k are adjacent
            int o = tok * stride + vec * dh + j;
            float v = vec < 2 * heads ? p.get(o) : 0f;
            part[tid] = v * v;
            ctx.localBarrier();
            for (int st = 64; st > 0; st >>= 1) {
                if (j < st) {
                    part[tid] += part[tid + st];
                }
                ctx.localBarrier();
            }
            float inv = 1f / TornadoMath.sqrt(part[seg * 128] / dh + eps);
            ctx.localBarrier();
            if (vec < 2 * heads) {
                p.set(o, v * inv);
            }
        }
    }

    /** Queries per work-group of {@link #variateAttention}. */
    public static final int ATTN_QUERIES = 16;

    /**
     * Attention across the rows of a group, per patch and head (flash style). One work-group of
     * 128 threads handles {@link #ATTN_QUERIES} consecutive kept rows (queries) for one (t, head):
     * each key's k and v vectors are loaded once into local memory and serve every query of the
     * block; thread {@code (q, lane)} forms 1/8 of query q's dot product, lane 0 applies the mask
     * and the online softmax, and thread j accumulates output dimension j for all queries. Keys
     * run over the union of the queries' group ranges only, so the cost is Σ group², not rows².
     *
     * <p>Rows are the kept rows {@code 0..nk}, token-major {@code [(row·seq + t)·stride]} holding
     * {@code q | k | v}; query q attends to keys {@code groupStart[q] .. groupEnd[q]} of its group,
     * except that a covariate query does not attend to target keys. Writes
     * {@code out [(row·seq + t)·heads·dh]}. Requires {@code dh == 128}.
     */
    public static void variateAttention(KernelContext ctx, FloatArray p, FloatArray out, IntArray groupStart, IntArray groupEnd,
            IntArray isTarget, int nk, int seq, int heads, int dh, int stride, HalfFloatArray th, int twin) {
        int wg = ctx.groupIdx;
        int tid = ctx.localIdx;
        int h = wg % heads;
        int t = (wg / heads) % seq;
        int q0 = (wg / (heads * seq)) * ATTN_QUERIES;
        int qn = TornadoMath.min(ATTN_QUERIES, nk - q0);
        int d = heads * dh;
        float scale = 1f / TornadoMath.sqrt(dh);
        float[] qs = ctx.allocateFloatLocalArray(ATTN_QUERIES * 128);
        float[] kv = ctx.allocateFloatLocalArray(256);
        float[] part = ctx.allocateFloatLocalArray(128);
        float[] mRun = ctx.allocateFloatLocalArray(ATTN_QUERIES);
        float[] lRun = ctx.allocateFloatLocalArray(ATTN_QUERIES);
        float[] corr = ctx.allocateFloatLocalArray(ATTN_QUERIES);
        float[] pw = ctx.allocateFloatLocalArray(ATTN_QUERIES);
        float[] acc = new float[ATTN_QUERIES];
        for (int qq = 0; qq < ATTN_QUERIES; qq++) {
            acc[qq] = 0f;
            if (qq < qn) {
                qs[qq * 128 + tid] = p.get(((q0 + qq) * seq + t) * stride + h * dh + tid);
            }
        }
        if (tid < ATTN_QUERIES) {
            mRun[tid] = -3.4e38f;
            lRun[tid] = 0f;
        }
        int kStart = groupStart.get(q0);
        int kEnd = groupEnd.get(q0 + qn - 1);
        int q = tid / 8;
        int lane = tid - q * 8;
        ctx.localBarrier();
        for (int k = kStart; k < kEnd; k++) {
            int ko = (k * seq + t) * stride + h * dh + tid;
            kv[tid] = p.get(ko + d);
            kv[128 + tid] = p.get(ko + 2 * d);
            ctx.localBarrier();
            float partial = 0f;
            for (int e = lane * 16; e < lane * 16 + 16; e++) {
                partial += qs[q * 128 + e] * kv[e];
            }
            part[tid] = partial;
            ctx.localBarrier();
            if (lane == 0 && q < qn) {
                float dot = 0f;
                for (int l = 0; l < 8; l++) {
                    dot += part[q * 8 + l];
                }
                int qi = q0 + q;
                // an int mask, not a short-circuit condition (works around a TornadoVM codegen issue)
                int inGroup = (k >= groupStart.get(qi) ? 1 : 0) * (k < groupEnd.get(qi) ? 1 : 0);
                int allowed = inGroup * (isTarget.get(qi) | (1 - isTarget.get(k)));
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
            float vj = kv[128 + tid];
            for (int qq = 0; qq < ATTN_QUERIES; qq++) {
                if (qq < qn) {
                    acc[qq] = acc[qq] * corr[qq] + pw[qq] * vj;
                }
            }
        }
        for (int qq = 0; qq < ATTN_QUERIES; qq++) {
            if (qq < qn) {
                int oi = ((q0 + qq) * seq + t) * d + h * dh + tid;
                float v = acc[qq] / lRun[qq];
                out.set(oi, v);
                if (twin == 1) {
                    th.set(oi, new HalfFloat(v));
                }
            }
        }
    }

    /** Threads of {@link #headNormAddRmsNorm}: one per element of a D = 512 row. */
    public static final int FUSED_NORM_THREADS = 512;

    /**
     * The end of an sLSTM time mixer and the start of its FFN, fused: per-head LayerNorm of the
     * cell output {@code x} (weight at {@code gOff}), added into the residual stream {@code h},
     * then RMSNorm of the new h (weight at {@code rOff}) into {@code n} and its FP16 twin. One
     * work-group of {@code heads·dh} threads per token, one element each; the per-head sums reduce
     * within each 128-thread segment and the RMS sum across the whole group.
     */
    public static void headNormAddRmsNorm(KernelContext ctx, FloatArray x, FloatArray h, FloatArray n, FloatArray w, int gOff,
            int rOff, int heads, int dh, float eps, float rmsEps, HalfFloatArray th, int twin) {
        int tok = ctx.groupIdx;
        int tid = ctx.localIdx;
        int d = heads * dh;
        int seg = tid / dh;
        int j = tid - seg * dh;
        float[] part = ctx.allocateFloatLocalArray(FUSED_NORM_THREADS);
        int idx = tok * d + tid;
        float v = x.get(idx);
        part[tid] = v;
        ctx.localBarrier();
        for (int st = 64; st > 0; st >>= 1) {
            if (j < st) {
                part[tid] += part[tid + st];
            }
            ctx.localBarrier();
        }
        float mean = part[seg * dh] / dh;
        ctx.localBarrier();
        part[tid] = (v - mean) * (v - mean);
        ctx.localBarrier();
        for (int st = 64; st > 0; st >>= 1) {
            if (j < st) {
                part[tid] += part[tid + st];
            }
            ctx.localBarrier();
        }
        float ln = (v - mean) / TornadoMath.sqrt(part[seg * dh] / dh + eps) * w.get(gOff + tid);
        float hv = h.get(idx) + ln;
        h.set(idx, hv);
        ctx.localBarrier();
        part[tid] = hv * hv;
        ctx.localBarrier();
        for (int st = FUSED_NORM_THREADS / 2; st > 0; st >>= 1) {
            if (tid < st) {
                part[tid] += part[tid + st];
            }
            ctx.localBarrier();
        }
        float out = hv / TornadoMath.sqrt(part[0] / d + rmsEps) * w.get(rOff + tid);
        n.set(idx, out);
        if (twin == 1) {
            th.set(idx, new HalfFloat(out));
        }
    }

    private TiRexKernels() {}
}
