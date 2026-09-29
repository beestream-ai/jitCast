package ai.beestream.jitrex.tornado;

import uk.ac.manchester.tornado.api.KernelContext;
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
            int rows, int len, int patch, int arcsinh, int binaryAware, float eps, HalfFloatArray th, int twin) {
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
            if (twin == 1) {
                th.set(o + i, new HalfFloat(here * z));
                th.set(o + patch + i, new HalfFloat(here));
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

    // ---- FP16 GEMMs written in TornadoVM (FP16 A and W, FP32 accumulation and C) -------------

    /** Output tile edge of {@link #gemmHalf}: a 16 × 16 work-group computes 64 × 64, 4 × 4 per thread. */
    public static final int HGEMM_TILE = 64;
    /** The K slice {@link #gemmHalf} stages in local memory. */
    public static final int HGEMM_K = 16;
    /** Threads per dimension of a {@link #gemmHalf} work-group. */
    public static final int HGEMM_THREADS = 16;

    /**
     * {@code C[m, n] (+)= act(A[m, :] · W[n, :] + b[n])} with FP16 {@code A} (the producer's
     * twin) and FP16 {@code W} ({@code [N, K]}), accumulated and written in FP32, plus the FP16
     * twin of {@code C} when {@code twin == 1}. The portable counterpart of
     * {@code cublasGemmExFP16FP32}: it runs on OpenCL and Metal as well as CUDA, without tensor
     * cores. Flags and epilogue order as {@link #gemm} (bias, SiLU, then accumulate).
     *
     * <p>Register-blocked: each work-group stages a 64 × 16 slice of A and of W in local memory
     * (converted to FP32 once), and each thread accumulates a 4 × 4 block whose rows and columns
     * are 16 apart, so the stores of a warp are contiguous. Global work
     * {@code (⌈N/64⌉·16, ⌈M/64⌉·16)}, local {@code (16, 16)}.
     */
    public static void gemmHalf(KernelContext ctx, HalfFloatArray a, HalfFloatArray wh, FloatArray c, FloatArray w, int bOff,
            int m, int n, int k, int flags, HalfFloatArray th, int twin) {
        int lx = ctx.localIdx;
        int ly = ctx.localIdy;
        int tid = ly * HGEMM_THREADS + lx;
        int rowBase = ctx.groupIdy * HGEMM_TILE;
        int colBase = ctx.groupIdx * HGEMM_TILE;
        float[] as = ctx.allocateFloatLocalArray(HGEMM_TILE * HGEMM_K);
        float[] ws = ctx.allocateFloatLocalArray(HGEMM_TILE * HGEMM_K);
        float c00 = 0f, c01 = 0f, c02 = 0f, c03 = 0f;
        float c10 = 0f, c11 = 0f, c12 = 0f, c13 = 0f;
        float c20 = 0f, c21 = 0f, c22 = 0f, c23 = 0f;
        float c30 = 0f, c31 = 0f, c32 = 0f, c33 = 0f;
        for (int kt = 0; kt < k; kt += HGEMM_K) {
            // 64 × 16 elements of each operand, 4 per thread; consecutive threads read along K
            for (int e = 0; e < 4; e++) {
                int idx = tid + e * HGEMM_THREADS * HGEMM_THREADS;
                int r = idx / HGEMM_K;
                int ka = kt + idx % HGEMM_K;
                float av = 0f;
                if (rowBase + r < m && ka < k) {
                    av = a.get((rowBase + r) * k + ka).getFloat32();
                }
                as[idx] = av;
                float wv = 0f;
                if (colBase + r < n && ka < k) {
                    wv = wh.get((colBase + r) * k + ka).getFloat32();
                }
                ws[idx] = wv;
            }
            ctx.localBarrier();
            for (int kk = 0; kk < HGEMM_K; kk++) {
                float a0 = as[ly * HGEMM_K + kk];
                float a1 = as[(ly + 16) * HGEMM_K + kk];
                float a2 = as[(ly + 32) * HGEMM_K + kk];
                float a3 = as[(ly + 48) * HGEMM_K + kk];
                float b0 = ws[lx * HGEMM_K + kk];
                float b1 = ws[(lx + 16) * HGEMM_K + kk];
                float b2 = ws[(lx + 32) * HGEMM_K + kk];
                float b3 = ws[(lx + 48) * HGEMM_K + kk];
                c00 += a0 * b0;
                c01 += a0 * b1;
                c02 += a0 * b2;
                c03 += a0 * b3;
                c10 += a1 * b0;
                c11 += a1 * b1;
                c12 += a1 * b2;
                c13 += a1 * b3;
                c20 += a2 * b0;
                c21 += a2 * b1;
                c22 += a2 * b2;
                c23 += a2 * b3;
                c30 += a3 * b0;
                c31 += a3 * b1;
                c32 += a3 * b2;
                c33 += a3 * b3;
            }
            ctx.localBarrier();
        }
        int r0 = rowBase + ly;
        int col0 = colBase + lx;
        gemmStore(c, w, bOff, r0, col0, m, n, flags, c00, th, twin);
        gemmStore(c, w, bOff, r0, col0 + 16, m, n, flags, c01, th, twin);
        gemmStore(c, w, bOff, r0, col0 + 32, m, n, flags, c02, th, twin);
        gemmStore(c, w, bOff, r0, col0 + 48, m, n, flags, c03, th, twin);
        gemmStore(c, w, bOff, r0 + 16, col0, m, n, flags, c10, th, twin);
        gemmStore(c, w, bOff, r0 + 16, col0 + 16, m, n, flags, c11, th, twin);
        gemmStore(c, w, bOff, r0 + 16, col0 + 32, m, n, flags, c12, th, twin);
        gemmStore(c, w, bOff, r0 + 16, col0 + 48, m, n, flags, c13, th, twin);
        gemmStore(c, w, bOff, r0 + 32, col0, m, n, flags, c20, th, twin);
        gemmStore(c, w, bOff, r0 + 32, col0 + 16, m, n, flags, c21, th, twin);
        gemmStore(c, w, bOff, r0 + 32, col0 + 32, m, n, flags, c22, th, twin);
        gemmStore(c, w, bOff, r0 + 32, col0 + 48, m, n, flags, c23, th, twin);
        gemmStore(c, w, bOff, r0 + 48, col0, m, n, flags, c30, th, twin);
        gemmStore(c, w, bOff, r0 + 48, col0 + 16, m, n, flags, c31, th, twin);
        gemmStore(c, w, bOff, r0 + 48, col0 + 32, m, n, flags, c32, th, twin);
        gemmStore(c, w, bOff, r0 + 48, col0 + 48, m, n, flags, c33, th, twin);
    }

    /** One output element of {@link #gemmHalf}: bias, SiLU, accumulate, store, FP16 twin. */
    private static void gemmStore(FloatArray c, FloatArray w, int bOff, int row, int col, int m, int n, int flags, float acc,
            HalfFloatArray th, int twin) {
        if (row < m && col < n) {
            float v = acc;
            if ((flags & BIAS) != 0) {
                v += w.get(bOff + col);
            }
            if ((flags & SILU) != 0) {
                v = v / (1f + TornadoMath.exp(-v));
            }
            int ci = row * n + col;
            if ((flags & ACCUMULATE) != 0) {
                v += c.get(ci);
            }
            c.set(ci, v);
            if (twin == 1) {
                th.set(ci, new HalfFloat(v));
            }
        }
    }

    /** Tile shape of {@link #gemmHalfTile}: 64 × 64 outputs per tile block, K in steps of 32. */
    public static final int CT_M = 64;
    public static final int CT_N = 64;
    public static final int CT_K = 32;

    /**
     * {@link #gemmHalf} with the CUDA Tile API (CUDA only, compute capability 8.0+, toolkit
     * 13.3+): one tile block per 64 × 64 output tile, {@code mma} on tensor cores with an FP32
     * accumulator, {@code W} read as {@code [N, K]} and transposed on load. Masked loads and stores
     * cover ragged M, N and K. The epilogue flags arrive as 0/1 floats and act through
     * {@code select}, so the only control flow is the scalar test around the FP16 twin store; a
     * kernel without a bias passes {@code bOff = 0} (any readable offset). The worker grid counts
     * tile blocks: {@code (⌈M/64⌉, ⌈N/64⌉)}.
     */
    public static void gemmHalfTile(TileContext tc, HalfFloatArray a, HalfFloatArray wh, FloatArray c, FloatArray w, int bOff,
            int m, int n, int k, float bias, float silu, float accumulate, HalfFloatArray th, int twin) {
        PartitionView av = tc.partition(tc.view(a, m, k), CT_M, CT_K);
        PartitionView wv = tc.partition(tc.view(wh, n, k), CT_N, CT_K);
        PartitionView cv = tc.partition(tc.view(c, m, n), CT_M, CT_N);
        PartitionView bv = tc.partition(tc.viewStrided(w, bOff, 1, n, n), 1, CT_N);
        int rowBlock = tc.bidX();
        int colBlock = tc.bidY();
        int kBlocks = (k + CT_K - 1) / CT_K;
        Tile acc = tc.zeros(DType.F32, CT_M, CT_N);
        for (int step = 0; step < kBlocks; step++) {
            acc = tc.mma(av.loadMasked(rowBlock, step), tc.transpose(wv.loadMasked(colBlock, step)), acc);
        }
        Tile zero = tc.zeros(DType.F32, CT_M, CT_N);
        Tile v = tc.select(tc.lessThan(zero, bias), tc.add(acc, bv.loadMasked(0, colBlock)), acc);
        Tile act = tc.div(v, tc.add(tc.ones(DType.F32, CT_M, CT_N), tc.exp(tc.scale(v, -1.0))));
        v = tc.select(tc.lessThan(zero, silu), act, v);
        v = tc.select(tc.lessThan(zero, accumulate), tc.add(v, cv.loadMasked(rowBlock, colBlock)), v);
        cv.storeMasked(v, rowBlock, colBlock);
        if (twin == 1) {
            PartitionView tv = tc.partition(tc.view(th, m, n), CT_M, CT_N);
            tv.storeMasked(tc.cast(v, DType.F16), rowBlock, colBlock);
        }
    }

    /** {@code y = x · rsqrt(mean(x²) + eps) · w} per row of {@code d}; one work-group per row. */
    public static void rmsNorm(KernelContext ctx, FloatArray x, FloatArray y, FloatArray w, int wOff, int d, float eps, HalfFloatArray th, int twin) {
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
            float v = x.get(base + i) * inv * w.get(wOff + i);
            y.set(base + i, v);
            if (twin == 1) {
                th.set(base + i, new HalfFloat(v));
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

    /** {@code out[i] = SiLU(big[r·2f + j]) · big[r·2f + f + j]} for the fused up-gate|up projection. */
    public static void swiGlu(KernelContext ctx, FloatArray big, FloatArray out, int n, int ffn, HalfFloatArray th, int twin) {
        int id = ctx.globalIdx;
        if (id < n * ffn) {
            int r = id / ffn;
            int j = id - r * ffn;
            float g = big.get(r * 2 * ffn + j);
            float u = big.get(r * 2 * ffn + ffn + j);
            float v = g / (1f + TornadoMath.exp(-g)) * u;
            out.set(id, v);
            if (twin == 1) {
                th.set(id, new HalfFloat(v));
            }
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

    /** {@code acc[i] += x[i]}. */
    public static void addInPlace(KernelContext ctx, FloatArray acc, FloatArray x, int n) {
        int i = ctx.globalIdx;
        if (i < n) {
            acc.set(i, acc.get(i) + x.get(i));
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
                // an int mask, not a short-circuit condition (TornadoVM codegen bug T1)
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
    public static void siluInPlace(KernelContext ctx, FloatArray x, int count, HalfFloatArray th, int twin) {
        int i = ctx.globalIdx;
        if (i < count) {
            float v = x.get(i);
            float y = v / (1f + TornadoMath.exp(-v));
            x.set(i, y);
            if (twin == 1) {
                th.set(i, new HalfFloat(y));
            }
        }
    }
    /** {@code dst[i] = (half) src[i]} for the first {@code count} elements (A operand of an FP16 GEMM). */
    public static void toHalf(KernelContext ctx, FloatArray src, HalfFloatArray dst, int count) {
        int i = ctx.globalIdx;
        if (i < count) {
            dst.set(i, new HalfFloat(src.get(i)));
        }
    }
    /**
     * After an all-FP16 GEMM: {@code C[i] = act(float(h[i]) + b[col]) (+ C[i])} with the
     * {@link #gemm} flags, so the FP32 residual stream only ever sees FP32 arithmetic.
     */
    public static void fromHalfEpilogue(KernelContext ctx, HalfFloatArray h, FloatArray c, FloatArray w, int bOff, int m, int n,
            int flags, HalfFloatArray th, int twin) {
        int i = ctx.globalIdx;
        if (i < m * n) {
            float v = h.get(i).getFloat32();
            if ((flags & BIAS) != 0) {
                v += w.get(bOff + i % n);
            }
            if ((flags & SILU) != 0) {
                v = v / (1f + TornadoMath.exp(-v));
            }
            if ((flags & ACCUMULATE) != 0) {
                v += c.get(i);
            }
            c.set(i, v);
            if (twin == 1) {
                th.set(i, new HalfFloat(v));
            }
        }
    }
    // ---- fused epilogues ------------------------------------------------------------------------

    /**
     * After a GEMM with β = 0: {@code C[i] = SiLU(C[i] + b[col])} (the bias when {@code bOff >= 0}),
     * and its FP16 twin — the bias fill, the SiLU and the conversion in one pass.
     */
    public static void biasSilu(KernelContext ctx, FloatArray c, FloatArray w, int bOff, int m, int n, HalfFloatArray th,
            int twin) {
        // a 1D launch: a 2D one (columns × rows) would exceed CUDA's 65,535 limit on gridDim.y
        // at large row counts
        int i = ctx.globalIdx;
        if (i < m * n) {
            int row = i / n;
            int col = i - row * n;
            float v = c.get(i);
            if (bOff >= 0) {
                v += w.get(bOff + col);
            }
            v = v / (1f + TornadoMath.exp(-v));
            c.set(i, v);
            if (twin == 1) {
                th.set(i, new HalfFloat(v));
            }
        }
    }

    /**
     * {@code h[r] += y[r]}, then {@code n[r] = RMSNorm(h[r])·w} (and its FP16 twin), one
     * work-group of {@link #NORM_THREADS} per row: the residual add and the following RMSNorm in
     * one pass over h.
     */
    public static void addRmsNorm(KernelContext ctx, FloatArray h, FloatArray y, FloatArray n, FloatArray w, int wOff, int d,
            float eps, HalfFloatArray th, int twin) {
        int row = ctx.groupIdx;
        int tid = ctx.localIdx;
        float[] part = ctx.allocateFloatLocalArray(NORM_THREADS);
        int base = row * d;
        float ss = 0f;
        for (int i = tid; i < d; i += NORM_THREADS) {
            float v = h.get(base + i) + y.get(base + i);
            h.set(base + i, v);
            ss += v * v;
        }
        part[tid] = ss;
        ctx.localBarrier();
        for (int st = NORM_THREADS / 2; st > 0; st >>= 1) {
            if (tid < st) {
                part[tid] += part[tid + st];
            }
            ctx.localBarrier();
        }
        float inv = 1f / TornadoMath.sqrt(part[0] / d + eps);
        for (int i = tid; i < d; i += NORM_THREADS) {
            float v = h.get(base + i) * inv * w.get(wOff + i);
            n.set(base + i, v);
            if (twin == 1) {
                th.set(base + i, new HalfFloat(v));
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
}
