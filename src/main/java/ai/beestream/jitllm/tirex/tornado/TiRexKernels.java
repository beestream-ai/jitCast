package ai.beestream.jitllm.tirex.tornado;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The TornadoVM kernels of the TiRex-2 forward pass (portable: OpenCL, Metal, CUDA). Each one is a
 * device version of a {@code cpu.*} stage and is tested against it.
 *
 * <p>All weights live in one packed {@link FloatArray} addressed by element offsets
 * ({@link PackedWeights}); activations are row-major {@code [rows · patches · width]}. Every kernel
 * is a {@link KernelContext} kernel launched with an explicit worker grid (no {@code @Parallel}
 * loops); element-wise ones map {@code ctx.globalIdx} to one item and bounds-check it.
 */
public final class TiRexKernels {

    /** GEMM tile edge; the work-group is {@code TILE × TILE}. */
    public static final int TILE = 16;

    /** {@link #gemm} flags. */
    public static final int BIAS = 1;
    public static final int ACCUMULATE = 2;
    public static final int SILU = 4;

    /** Work-group width of {@link #rmsNorm}. */
    public static final int NORM_THREADS = 256;

    private TiRexKernels() {}

    /**
     * Scaler + patching ({@code cpu.InputStage}): one work-group of {@link #NORM_THREADS} threads
     * per row, with the sums reduced in local memory. The host uploads the values with NaN replaced
     * by 0 in {@code x} and the is-present flags in {@code present}: device compilers may assume
     * there are no NaNs (Metal's default fast math does), so a {@code v != v} test is not portable.
     * Writes the embedding input {@code tok [(r·patches + t)·2P + i]} (values, then the is-present
     * mask) and the scaler state {@code stats [r·3 + {0: loc, 1: scale, 2: binary}]}.
     */
    public static void scalePatch(KernelContext ctx, FloatArray x, FloatArray present, FloatArray tok, FloatArray stats,
            int rows, int len, int patch, int arcsinh, int binaryAware, float eps) {
        int r = ctx.groupIdx;
        int tid = ctx.localIdx;
        float[] sum = ctx.allocateFloatLocalArray(NORM_THREADS);
        float[] cnt = ctx.allocateFloatLocalArray(NORM_THREADS);
        float[] nonBinary = ctx.allocateFloatLocalArray(NORM_THREADS);
        int base = r * len;
        float s0 = 0f, c0 = 0f, nb = 0f;
        for (int t = tid; t < len; t += NORM_THREADS) {
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
        for (int st = NORM_THREADS / 2; st > 0; st >>= 1) {
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
        for (int t = tid; t < len; t += NORM_THREADS) {
            float d = x.get(base + t) - loc;
            ss += present.get(base + t) * d * d;
        }
        sum[tid] = ss;
        ctx.localBarrier();
        for (int st = NORM_THREADS / 2; st > 0; st >>= 1) {
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
        for (int pos = tid; pos < len; pos += NORM_THREADS) {
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
        }
    }

    /**
     * {@code C[m, n] (+)= act(A[m, :] · W[n, :] + b[n])}, tiled in local memory. {@code W} is
     * PyTorch {@code [N, K]} at {@code wOff} in the packed weights; {@code b} at {@code bOff}.
     * Global work {@code (⌈N/T⌉·T, ⌈M/T⌉·T)}, local {@code (T, T)}.
     */
    public static void gemm(KernelContext ctx, FloatArray a, int aOff, FloatArray w, int wOff, int bOff, FloatArray c,
            int cOff, int m, int n, int k, int flags) {
        int lx = ctx.localIdx;
        int ly = ctx.localIdy;
        int col = ctx.groupIdx * TILE + lx;
        int row = ctx.groupIdy * TILE + ly;
        int colBase = ctx.groupIdx * TILE;
        float[] as = ctx.allocateFloatLocalArray(TILE * TILE);
        float[] ws = ctx.allocateFloatLocalArray(TILE * TILE);
        float acc = 0f;
        for (int kt = 0; kt < k; kt += TILE) {
            int ka = kt + lx;
            as[ly * TILE + lx] = (row < m && ka < k) ? a.get(aOff + row * k + ka) : 0f;
            int wr = colBase + ly;
            ws[ly * TILE + lx] = (wr < n && ka < k) ? w.get(wOff + wr * k + ka) : 0f;
            ctx.localBarrier();
            for (int j = 0; j < TILE; j++) {
                acc += as[ly * TILE + j] * ws[lx * TILE + j];
            }
            ctx.localBarrier();
        }
        if (row < m && col < n) {
            if ((flags & BIAS) != 0) {
                acc += w.get(bOff + col);
            }
            if ((flags & SILU) != 0) {
                acc = acc / (1f + TornadoMath.exp(-acc));
            }
            int ci = cOff + row * n + col;
            if ((flags & ACCUMULATE) != 0) {
                acc += c.get(ci);
            }
            c.set(ci, acc);
        }
    }

    /** {@code y = x · rsqrt(mean(x²) + eps) · w} per row of {@code d}; one work-group per row. */
    public static void rmsNorm(KernelContext ctx, FloatArray x, FloatArray y, FloatArray w, int wOff, int d, float eps) {
        int row = ctx.groupIdx;
        int tid = ctx.localIdx;
        float[] part = ctx.allocateFloatLocalArray(NORM_THREADS);
        int base = row * d;
        float ss = 0f;
        for (int i = tid; i < d; i += NORM_THREADS) {
            float v = x.get(base + i);
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
            y.set(base + i, x.get(base + i) * inv * w.get(wOff + i));
        }
    }

    /**
     * Per-head LayerNorm (centred, biased variance, full-width weight), one work-group of
     * {@code dh} threads per (row, head). {@code mode} 0: {@code y = LN(x)}; 1: {@code y = σ(g) ·
     * LN(x)} with the gate read at {@code gOff + rowIndex·gStride}; 2: {@code y += LN(x)}.
     */
    public static void headLayerNorm(KernelContext ctx, FloatArray x, FloatArray y, FloatArray w, int wOff, int heads,
            int dh, float eps, FloatArray g, int gStride, int gOff, int mode) {
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
    }

    /** The mLSTM kernel keeps one step's scores in local memory: sequences up to this length. */
    public static final int MLSTM_MAX_SEQ = 128;

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
        // logσ(capped f_s) and capped i_s for s ≤ t
        if (j <= t) {
            int so = (r * seq + j) * stride;
            float f = p.get(so + fOff + h);
            f = cap * TornadoMath.tanh(f / cap);
            cum[j] = TornadoMath.min(f, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(f)));
            float ig = p.get(so + iOff + h);
            logD[j] = cap * TornadoMath.tanh(ig / cap);
        }
        ctx.localBarrier();
        if (j == 0) {
            float acc = 0f;
            for (int s = 0; s <= t; s++) {
                acc += cum[s];
                cum[s] = acc;
            }
        }
        ctx.localBarrier();
        float ft = cum[t];
        if (j <= t) {
            // logD[t, s] = F[t] − F[s] + i[s], and the scaled score q_t·k_s
            logD[j] = ft - cum[j] + logD[j];
            int qo = (r * seq + t) * stride + h * dqk;
            int ko = (r * seq + j) * stride + kOff + h * dqk;
            float dot = 0f;
            for (int e = 0; e < dqk; e++) {
                dot += p.get(qo + e) * p.get(ko + e);
            }
            score[j] = dot * scale;
        }
        ctx.localBarrier();
        // m[t] = max(F[t], max_s logD[t, s]), computed redundantly by every thread
        float m = ft;
        for (int s = 0; s <= t; s++) {
            m = TornadoMath.max(m, logD[s]);
        }
        ctx.localBarrier();
        if (j <= t) {
            score[j] = score[j] * TornadoMath.exp(logD[j] - m);
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

    /**
     * The sLSTM recurrence ({@code cpu.Cells.slstmRecurrence}): one work-group per (row, head) runs
     * all {@code seq} steps; heads are independent because R is block-diagonal, so no work-group
     * waits on another. The {@code 4·dh / gatesPerThread} threads split the four gates: thread
     * {@code (gBlock, e)} computes gates {@code gBlock·gpt …} of lane e, whose recurrent sum reads
     * one R column per gate (coalesced over e); the raw gates go to local memory, and threads
     * {@code e < dh} then update the stabilised state and publish y for the next step.
     */
    public static void slstmRecurrence(KernelContext ctx, FloatArray wx, FloatArray w, int rOff, int bOff, FloatArray y,
            int seq, int heads, int gatesPerThread) {
        int rh = ctx.groupIdx;
        int tid = ctx.localIdx;
        int row = rh / heads;
        int h = rh - row * heads;
        int dh = 128;
        int d = heads * dh;
        int e = tid % dh;
        int g0 = (tid / dh) * gatesPerThread;
        float[] yl = ctx.allocateFloatLocalArray(128);
        float[] raw = ctx.allocateFloatLocalArray(512);
        if (tid < dh) {
            yl[tid] = 0f;
        }
        float c = 0f;
        float nn = 0f;
        float m = 0f;
        ctx.localBarrier();
        for (int t = 0; t < seq; t++) {
            int base = (row * seq + t) * 4 * d + h * dh + e;
            for (int gi = 0; gi < gatesPerThread; gi++) {
                int g = g0 + gi;
                float acc = wx.get(base + g * d) + w.get(bOff + (h * 4 + g) * dh + e);
                int ro = rOff + (h * dh * 4 + g) * dh + e;
                for (int pp = 0; pp < dh; pp++) {
                    acc += yl[pp] * w.get(ro + pp * 4 * dh);
                }
                raw[g * dh + e] = acc;
            }
            ctx.localBarrier();
            if (tid < dh) {
                float ri = raw[e];
                float rf = raw[dh + e];
                float rz = raw[2 * dh + e];
                float ro = raw[3 * dh + e];
                float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m;
                float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
                float ig = TornadoMath.exp(ri - mNew);
                float fg = TornadoMath.exp(logf - mNew);
                c = fg * c + ig * TornadoMath.tanh(rz);
                nn = TornadoMath.max(fg * nn + ig, 1f);
                float yNew = c / nn / (1f + TornadoMath.exp(-ro));
                m = mNew;
                yl[e] = yNew;
                y.set((row * seq + t) * d + h * dh + e, yNew);
            }
            ctx.localBarrier();
        }
    }

    /** {@code out[i] = SiLU(big[r·2f + j]) · big[r·2f + f + j]} for the fused up-gate|up projection. */
    public static void swiGlu(KernelContext ctx, FloatArray big, FloatArray out, int n, int ffn) {
        int id = ctx.globalIdx;
        if (id < n * ffn) {
            int r = id / ffn;
            int j = id - r * ffn;
            float g = big.get(r * 2 * ffn + j);
            float u = big.get(r * 2 * ffn + ffn + j);
            out.set(id, g / (1f + TornadoMath.exp(-g)) * u);
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
    public static void gatherRows(KernelContext ctx, FloatArray src, FloatArray dst, IntArray idx, int count, int seq, int d, int flip) {
        int id = ctx.globalIdx;
        if (id < count * seq * d) {
            int k = id / (seq * d);
            int rem = id - k * seq * d;
            int t = rem / d;
            int j = rem - t * d;
            int ts = flip == 1 ? seq - 1 - t : t;
            dst.set(id, src.get((idx.get(k) * seq + ts) * d + j));
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
    public static void concatForward(KernelContext ctx, FloatArray y, FloatArray rev, IntArray known, FloatArray cat, int count, int seq, int d) {
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
        }
    }

    /** {@code acc[i] += x[i]}. */
    public static void addInPlace(KernelContext ctx, FloatArray acc, FloatArray x, int n) {
        int i = ctx.globalIdx;
        if (i < n) {
            acc.set(i, acc.get(i) + x.get(i));
        }
    }

    /**
     * RMS-normalises q and k of the fused {@code q | k | v} projection in place, per head, with no
     * weight ({@code F.rms_norm}); one thread per (token, q or k, head).
     */
    public static void qkNorm(KernelContext ctx, FloatArray p, int tokens, int heads, int dh, int stride, float eps) {
        int id = ctx.globalIdx;
        if (id < tokens * 2 * heads) {
            int tok = id / (2 * heads);
            int rem = id - tok * 2 * heads;
            int o = tok * stride + rem * dh; // rem = which·heads + h, and q | k are adjacent
            float ss = 0f;
            for (int j = 0; j < dh; j++) {
                float v = p.get(o + j);
                ss += v * v;
            }
            float inv = 1f / TornadoMath.sqrt(ss / dh + eps);
            for (int j = 0; j < dh; j++) {
                p.set(o + j, p.get(o + j) * inv);
            }
        }
    }

    /**
     * Attention across the rows of a group, per patch and head; one thread per (query row, t, head).
     * Rows are the kept rows {@code 0..nk}, token-major {@code [(row·seq + t)·stride]}; a query
     * attends to the keys {@code groupStart[q] .. groupEnd[q]} of its own group, except that a
     * covariate query does not attend to target keys. Writes {@code out [(row·seq + t)·heads·dh]}.
     */
    public static void variateAttention(KernelContext ctx, FloatArray p, FloatArray out, IntArray groupStart, IntArray groupEnd,
            IntArray isTarget, int nk, int seq, int heads, int dh, int stride) {
        int id = ctx.globalIdx;
        if (id < nk * seq * heads) {
            int q = id / (seq * heads);
            int rem = id - q * seq * heads;
            int t = rem / heads;
            int h = rem - t * heads;
            int d = heads * dh;
            float scale = 1f / TornadoMath.sqrt(dh);
            int qo = (q * seq + t) * stride + h * dh;
            int qTarget = isTarget.get(q);
            int s0 = groupStart.get(q);
            int s1 = groupEnd.get(q);
            float max = -3.4e38f;
            // The mask is an int, not `qTarget == 1 || isTarget[k] == 0`: TornadoVM 7.0.1 emits that
            // short-circuit inside a loop without a `continue`, so a masked key falls through into the
            // loop body with the next k.
            for (int k = s0; k < s1; k++) {
                int allowed = qTarget | (1 - isTarget.get(k));
                if (allowed != 0) {
                    int ko = (k * seq + t) * stride + d + h * dh;
                    float dot = 0f;
                    for (int j = 0; j < dh; j++) {
                        dot += p.get(qo + j) * p.get(ko + j);
                    }
                    max = TornadoMath.max(max, dot * scale);
                }
            }
            float[] acc = new float[128];
            for (int j = 0; j < dh; j++) {
                acc[j] = 0f;
            }
            float sum = 0f;
            for (int k = s0; k < s1; k++) {
                int allowed = qTarget | (1 - isTarget.get(k));
                if (allowed != 0) {
                    int ko = (k * seq + t) * stride + d + h * dh;
                    float dot = 0f;
                    for (int j = 0; j < dh; j++) {
                        dot += p.get(qo + j) * p.get(ko + j);
                    }
                    float e = TornadoMath.exp(dot * scale - max);
                    sum += e;
                    int vo = (k * seq + t) * stride + 2 * d + h * dh;
                    for (int j = 0; j < dh; j++) {
                        acc[j] += e * p.get(vo + j);
                    }
                }
            }
            int oo = (q * seq + t) * d + h * dh;
            for (int j = 0; j < dh; j++) {
                out.set(oo + j, acc[j] / sum);
            }
        }
    }
    // ---- CUDA hybrid path: epilogues around a cuBLAS GEMM -----------------------------------------

    /** Before a cuBLAS GEMM with β = 1: {@code C[r, c] = (accumulate ? C[r, c] : 0) + b[c]}. */
    public static void biasFill(KernelContext ctx, FloatArray c, FloatArray w, int bOff, int m, int n, int accumulate) {
        int id = ctx.globalIdx;
        if (id < m * n) {
            int col = id % n;
            float base = accumulate == 1 ? c.get(id) : 0f;
            c.set(id, base + w.get(bOff + col));
        }
    }

    /** {@code x[i] = SiLU(x[i])} for the first {@code count} elements. */
    public static void siluInPlace(KernelContext ctx, FloatArray x, int count) {
        int i = ctx.globalIdx;
        if (i < count) {
            float v = x.get(i);
            x.set(i, v / (1f + TornadoMath.exp(-v)));
        }
    }
}
