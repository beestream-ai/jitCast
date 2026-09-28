package ai.beestream.jitllm.tirex.tornado;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * The TornadoVM kernels of the TiRex-2 forward pass (portable: OpenCL, Metal, CUDA). Each one is a
 * device version of a {@code cpu.*} stage and is tested against it.
 *
 * <p>All weights live in one packed {@link FloatArray} addressed by element offsets
 * ({@link PackedWeights}); activations are row-major {@code [rows · patches · width]}.
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
     * Scaler + patching ({@code cpu.InputStage}), one thread per row. The host uploads the values
     * with NaN replaced by 0 in {@code x} and the is-present flags in {@code present}: device
     * compilers may assume there are no NaNs (Metal's default fast math does), so a {@code v != v}
     * test is not portable. Writes the embedding input
     * {@code tok [(r·patches + t)·2P + i]} (values, then the is-present mask) and the scaler state
     * {@code stats [r·3 + {0: loc, 1: scale, 2: binary}]}.
     */
    public static void scalePatch(FloatArray x, FloatArray present, FloatArray tok, FloatArray stats, int rows, int len,
            int patch, int arcsinh, int binaryAware, float eps) {
        for (@uk.ac.manchester.tornado.api.annotations.Parallel int r = 0; r < rows; r++) {
            int base = r * len;
            float sum = 0f;
            int n = 0;
            int binary = 1;
            for (int t = 0; t < len; t++) {
                float v = x.get(base + t);
                if (present.get(base + t) != 0f) {
                    sum += v;
                    n++;
                    if (v != 0f && v != 1f) {
                        binary = 0;
                    }
                }
            }
            float loc = n == 0 ? 0f : sum / n;
            float ss = 0f;
            for (int t = 0; t < len; t++) {
                float v = x.get(base + t);
                if (present.get(base + t) != 0f) {
                    ss += (v - loc) * (v - loc);
                }
            }
            float scale = n == 0 ? 1f : TornadoMath.sqrt(ss / n);
            scale = TornadoMath.max(scale, eps);
            int isBinary = binaryAware * binary;
            if (isBinary == 1) {
                loc = 0f;
                scale = 1f;
            }
            stats.set(r * 3, loc);
            stats.set(r * 3 + 1, scale);
            stats.set(r * 3 + 2, isBinary);
            int doAsinh = arcsinh * (1 - isBinary);
            int patches = len / patch;
            for (int t = 0; t < patches; t++) {
                int o = (r * patches + t) * 2 * patch;
                for (int i = 0; i < patch; i++) {
                    float v = x.get(base + t * patch + i);
                    float z = (v - loc) / scale;
                    if (doAsinh == 1) {
                        z = TornadoMath.asinh(z);
                    }
                    float here = present.get(base + t * patch + i);
                    tok.set(o + i, here != 0f ? z : 0f);
                    tok.set(o + patch + i, here);
                }
            }
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

    /**
     * The mLSTM cell (parallel stabilised form, {@code cpu.Cells.mlstm}), one thread per
     * (row, head, t). Reads q, k, v and the i/f pre-activations from the fused projection
     * {@code p [(r·seq + t)·stride + …]} and writes {@code h [(r·seq + t)·heads·dv + head·dv + j]}.
     */
    public static void mlstmCell(FloatArray p, FloatArray hOut, int rows, int seq, int heads, int stride,
            int kOff, int vOff, int iOff, int fOff, float cap, float eps) {
        for (@uk.ac.manchester.tornado.api.annotations.Parallel int id = 0; id < rows * heads * seq; id++) {
            int t = id % seq;
            int rh = id / seq;
            int h = rh % heads;
            int r = rh / heads;
            int dqk = 64;
            int dv = 128;
            float scale = 0.125f; // 1/sqrt(64)
            // F[t] and the stabiliser m[t] = max(F[t], max_s F[t] − F[s] + i[s])
            float ft = 0f;
            for (int s = 0; s <= t; s++) {
                float f = p.get((r * seq + s) * stride + fOff + h);
                f = cap * TornadoMath.tanh(f / cap);
                ft += TornadoMath.min(f, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(f)));
            }
            float m = ft;
            float fs = 0f;
            for (int s = 0; s <= t; s++) {
                float f = p.get((r * seq + s) * stride + fOff + h);
                f = cap * TornadoMath.tanh(f / cap);
                fs += TornadoMath.min(f, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(f)));
                float ig = p.get((r * seq + s) * stride + iOff + h);
                ig = cap * TornadoMath.tanh(ig / cap);
                m = TornadoMath.max(m, ft - fs + ig);
            }
            float[] num = new float[128];
            for (int j = 0; j < dv; j++) {
                num[j] = 0f;
            }
            float den = 0f;
            int qo = (r * seq + t) * stride + h * dqk;
            fs = 0f;
            for (int s = 0; s <= t; s++) {
                int so = (r * seq + s) * stride;
                float f = p.get(so + fOff + h);
                f = cap * TornadoMath.tanh(f / cap);
                fs += TornadoMath.min(f, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(f)));
                float ig = p.get(so + iOff + h);
                ig = cap * TornadoMath.tanh(ig / cap);
                float dot = 0f;
                for (int j = 0; j < dqk; j++) {
                    dot += p.get(qo + j) * p.get(so + kOff + h * dqk + j);
                }
                float cts = dot * scale * TornadoMath.exp(ft - fs + ig - m);
                den += cts;
                for (int j = 0; j < dv; j++) {
                    num[j] += cts * p.get(so + vOff + h * dv + j);
                }
            }
            float norm = TornadoMath.max(TornadoMath.abs(den), TornadoMath.exp(-m)) + eps;
            int ho = (r * seq + t) * heads * dv + h * dv;
            for (int j = 0; j < dv; j++) {
                hOut.set(ho + j, num[j] / norm);
            }
        }
    }

    /**
     * The four headwise sLSTM gate projections, one thread per output:
     * {@code out[rt·4d + g·d + h·dh + o] = Σ_j x[rt·d + h·dh + j] · W_g[h][o][j]}.
     */
    public static void headwiseGates(FloatArray x, FloatArray w, int wOff, FloatArray out, int n, int heads, int dh) {
        int d = heads * dh;
        for (@uk.ac.manchester.tornado.api.annotations.Parallel int id = 0; id < n * 4 * d; id++) {
            int rt = id / (4 * d);
            int rem = id - rt * 4 * d;
            int g = rem / d;
            int hd = rem - g * d;
            int h = hd / dh;
            int o = hd - h * dh;
            int wo = wOff + ((g * heads + h) * dh + o) * dh;
            int xo = rt * d + h * dh;
            float acc = 0f;
            for (int j = 0; j < dh; j++) {
                acc += x.get(xo + j) * w.get(wo + j);
            }
            out.set(id, acc);
        }
    }

    /**
     * The sLSTM recurrence ({@code cpu.Cells.slstm}): one work-group of {@code dh = 128} threads
     * per (row, head) runs all {@code seq} steps; y is shared in local memory. Heads are independent
     * because R is block-diagonal, so no work-group ever waits on another.
     */
    public static void slstmRecurrence(KernelContext ctx, FloatArray wx, FloatArray w, int rOff, int bOff, FloatArray y,
            int seq, int heads) {
        int rh = ctx.groupIdx;
        int e = ctx.localIdx;
        int row = rh / heads;
        int h = rh - row * heads;
        int dh = 128;
        int d = heads * dh;
        float[] yl = ctx.allocateFloatLocalArray(128);
        yl[e] = 0f;
        float c = 0f;
        float nn = 0f;
        float m = 0f;
        float bi = w.get(bOff + (h * 4) * dh + e);
        float bf = w.get(bOff + (h * 4 + 1) * dh + e);
        float bz = w.get(bOff + (h * 4 + 2) * dh + e);
        float bo = w.get(bOff + (h * 4 + 3) * dh + e);
        ctx.localBarrier();
        for (int t = 0; t < seq; t++) {
            int base = (row * seq + t) * 4 * d + h * dh + e;
            float ri = wx.get(base) + bi;
            float rf = wx.get(base + d) + bf;
            float rz = wx.get(base + 2 * d) + bz;
            float ro = wx.get(base + 3 * d) + bo;
            for (int p = 0; p < dh; p++) {
                float yp = yl[p];
                int ro4 = rOff + ((h * dh + p) * 4) * dh + e;
                ri += yp * w.get(ro4);
                rf += yp * w.get(ro4 + dh);
                rz += yp * w.get(ro4 + 2 * dh);
                ro += yp * w.get(ro4 + 3 * dh);
            }
            float logf = TornadoMath.min(rf, 0f) - TornadoMath.log(1f + TornadoMath.exp(-TornadoMath.abs(rf))) + m;
            float mNew = t == 0 ? ri : TornadoMath.max(ri, logf);
            float ig = TornadoMath.exp(ri - mNew);
            float fg = TornadoMath.exp(logf - mNew);
            c = fg * c + ig * TornadoMath.tanh(rz);
            nn = TornadoMath.max(fg * nn + ig, 1f);
            float yNew = c / nn / (1f + TornadoMath.exp(-ro));
            m = mNew;
            ctx.localBarrier();
            yl[e] = yNew;
            y.set((row * seq + t) * d + h * dh + e, yNew);
            ctx.localBarrier();
        }
    }

    /** {@code out[i] = SiLU(big[r·2f + j]) · big[r·2f + f + j]} for the fused up-gate|up projection. */
    public static void swiGlu(FloatArray big, FloatArray out, int n, int ffn) {
        for (@uk.ac.manchester.tornado.api.annotations.Parallel int id = 0; id < n * ffn; id++) {
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
    public static void unpatch(FloatArray y, FloatArray stats, FloatArray out, int rows, int patches, int nq,
            int patch, int arcsinh, float clamp) {
        int len = patches * patch;
        for (@uk.ac.manchester.tornado.api.annotations.Parallel int id = 0; id < rows * nq * len; id++) {
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
}
