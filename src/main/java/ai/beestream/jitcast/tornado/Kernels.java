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
package ai.beestream.jitcast.tornado;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.tile.DType;
import uk.ac.manchester.tornado.api.tile.PartitionView;
import uk.ac.manchester.tornado.api.tile.Tile;
import uk.ac.manchester.tornado.api.tile.TileContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * TornadoVM kernels shared by every model family (portable: OpenCL, Metal, CUDA): the GEMMs, RMSNorm,
 * SwiGLU and the element-wise epilogues around them. Each one is a device version of a
 * {@code cpu.Ops} operator and is tested against it.
 *
 * <p>Weights live in one packed {@link FloatArray} addressed by element offsets; activations are
 * row-major {@code [rows · width]}. Every kernel is a {@link KernelContext} kernel launched with an
 * explicit worker grid (no {@code @Parallel} loops), except {@link #gemmHalfTile}, which uses
 * TileContext; element-wise ones map {@code ctx.globalIdx} to one item and bounds-check it.
 */
public final class Kernels {

    /** GEMM tile edge; the work-group is {@code TILE × TILE}. */
    public static final int TILE = 16;
    /** {@link #gemm} flags. */
    public static final int BIAS = 1;
    public static final int ACCUMULATE = 2;
    public static final int SILU = 4;
    /** {@link #gemmHalf}: write only the FP16 twin of C, not C (no bias, SiLU or accumulation). */
    public static final int TWIN_ONLY = 8;
    /** {@link #gemm} flag: ReLU instead of SiLU after the bias. */
    public static final int RELU = 16;
    /** Work-group width of {@link #rmsNorm}. */
    public static final int NORM_THREADS = 256;
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
        // W's tile K-major with a padded pitch: conflict-free reads across lx
        float[] ws = ctx.allocateFloatLocalArray((TILE + 1) * TILE);
        float acc = 0f;
        for (int kt = 0; kt < k; kt += TILE) {
            int ka = kt + lx;
            as[ly * TILE + lx] = (row < m && ka < k) ? a.get(aOff + row * k + ka) : 0f;
            int wr = colBase + ly;
            ws[lx * (TILE + 1) + ly] = (wr < n && ka < k) ? w.get(wOff + wr * k + ka) : 0f;
            ctx.localBarrier();
            for (int j = 0; j < TILE; j++) {
                acc += as[ly * TILE + j] * ws[j * (TILE + 1) + lx];
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
            if ((flags & RELU) != 0) {
                acc = TornadoMath.max(acc, 0f);
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
     * Row pitch of the staged slices: K-major, one padding element per row, so the threads of a
     * SIMD group read consecutive local addresses and store without bank conflicts.
     */
    static final int HGEMM_PITCH = HGEMM_TILE + 1;
    /**
     * {@code C[m, n] (+)= act(A[m, :] · W[n, :] + b[n])} with FP16 {@code A} (the producer's
     * twin) and FP16 {@code W} ({@code [N, K]}), accumulated and written in FP32, plus the FP16
     * twin of {@code C} when {@code twin == 1}. The portable counterpart of
     * {@code cublasGemmExFP16FP32}: it runs on OpenCL and Metal as well as CUDA, without tensor
     * cores. Flags and epilogue order as {@link #gemm} (bias, SiLU or ReLU, then accumulate).
     *
     * <p>Register-blocked: each work-group stages a 64 × 16 slice of A and of W in local memory
     * (converted to FP32 once, K-major with a padded pitch), and each thread accumulates a 4 × 4
     * block whose rows and columns are 16 apart, so the stores of a warp are contiguous. Global work
     * {@code (⌈N/64⌉·16, ⌈M/64⌉·16)}, local {@code (16, 16)}.
     */
    public static void gemmHalf(KernelContext ctx, HalfFloatArray a, HalfFloatArray wh, FloatArray c, FloatArray w, int bOff,
            int m, int n, int k, int flags, HalfFloatArray th, int twin) {
        int lx = ctx.localIdx;
        int ly = ctx.localIdy;
        int tid = ly * HGEMM_THREADS + lx;
        int rowBase = ctx.groupIdy * HGEMM_TILE;
        int colBase = ctx.groupIdx * HGEMM_TILE;
        float[] as = ctx.allocateFloatLocalArray(HGEMM_PITCH * HGEMM_K);
        float[] ws = ctx.allocateFloatLocalArray(HGEMM_PITCH * HGEMM_K);
        float c00 = 0f, c01 = 0f, c02 = 0f, c03 = 0f;
        float c10 = 0f, c11 = 0f, c12 = 0f, c13 = 0f;
        float c20 = 0f, c21 = 0f, c22 = 0f, c23 = 0f;
        float c30 = 0f, c31 = 0f, c32 = 0f, c33 = 0f;
        for (int kt = 0; kt < k; kt += HGEMM_K) {
            // 64 × 16 elements of each operand, 4 per thread; consecutive threads read along K
            for (int e = 0; e < 4; e++) {
                int idx = tid + e * HGEMM_THREADS * HGEMM_THREADS;
                int r = idx / HGEMM_K;
                int kk0 = idx % HGEMM_K;
                int ka = kt + kk0;
                float av = 0f;
                if (rowBase + r < m && ka < k) {
                    av = a.get((rowBase + r) * k + ka).getFloat32();
                }
                as[kk0 * HGEMM_PITCH + r] = av;
                float wv = 0f;
                if (colBase + r < n && ka < k) {
                    wv = wh.get((colBase + r) * k + ka).getFloat32();
                }
                ws[kk0 * HGEMM_PITCH + r] = wv;
            }
            ctx.localBarrier();
            for (int kk = 0; kk < HGEMM_K; kk++) {
                int o = kk * HGEMM_PITCH;
                float a0 = as[o + ly];
                float a1 = as[o + ly + 16];
                float a2 = as[o + ly + 32];
                float a3 = as[o + ly + 48];
                float b0 = ws[o + lx];
                float b1 = ws[o + lx + 16];
                float b2 = ws[o + lx + 32];
                float b3 = ws[o + lx + 48];
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
        if (row < m && col < n && (flags & TWIN_ONLY) != 0) {
            th.set(row * n + col, new HalfFloat(acc));
        } else if (row < m && col < n) {
            float v = acc;
            if ((flags & BIAS) != 0) {
                v += w.get(bOff + col);
            }
            if ((flags & SILU) != 0) {
                v = v / (1f + TornadoMath.exp(-v));
            }
            if ((flags & RELU) != 0) {
                v = TornadoMath.max(v, 0f);
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
    /**
     * {@link #gemm} register-blocked as {@link #gemmHalf} is, in FP32 throughout: {@code C[m, n] (+)=
     * act(A[m, :] · W[n, :] + b[n])} with {@code A} at {@code aOff}, {@code W} ({@code [N, K]}) at
     * {@code wOff} and {@code b} at {@code bOff} in the packed weights, and {@code C} at {@code cOff}.
     * Each work-group stages a 64 × 16 slice of A and of W in local memory, and each thread
     * accumulates a 4 × 4 block whose rows and columns are 16 apart. Flags and epilogue order as
     * {@link #gemm}. Global work {@code (⌈N/64⌉·16, ⌈M/64⌉·16)}, local {@code (16, 16)}.
     */
    public static void gemmBlocked(KernelContext ctx, FloatArray a, int aOff, FloatArray w, int wOff, int bOff, FloatArray c,
            int cOff, int m, int n, int k, int flags) {
        int lx = ctx.localIdx;
        int ly = ctx.localIdy;
        int tid = ly * HGEMM_THREADS + lx;
        int rowBase = ctx.groupIdy * HGEMM_TILE;
        int colBase = ctx.groupIdx * HGEMM_TILE;
        float[] as = ctx.allocateFloatLocalArray(HGEMM_PITCH * HGEMM_K);
        float[] ws = ctx.allocateFloatLocalArray(HGEMM_PITCH * HGEMM_K);
        float c00 = 0f, c01 = 0f, c02 = 0f, c03 = 0f;
        float c10 = 0f, c11 = 0f, c12 = 0f, c13 = 0f;
        float c20 = 0f, c21 = 0f, c22 = 0f, c23 = 0f;
        float c30 = 0f, c31 = 0f, c32 = 0f, c33 = 0f;
        for (int kt = 0; kt < k; kt += HGEMM_K) {
            // 64 × 16 elements of each operand, 4 per thread; consecutive threads read along K
            for (int e = 0; e < 4; e++) {
                int idx = tid + e * HGEMM_THREADS * HGEMM_THREADS;
                int r = idx / HGEMM_K;
                int kk0 = idx % HGEMM_K;
                int ka = kt + kk0;
                float av = 0f;
                if (rowBase + r < m && ka < k) {
                    av = a.get(aOff + (rowBase + r) * k + ka);
                }
                as[kk0 * HGEMM_PITCH + r] = av;
                float wv = 0f;
                if (colBase + r < n && ka < k) {
                    wv = w.get(wOff + (colBase + r) * k + ka);
                }
                ws[kk0 * HGEMM_PITCH + r] = wv;
            }
            ctx.localBarrier();
            for (int kk = 0; kk < HGEMM_K; kk++) {
                int o = kk * HGEMM_PITCH;
                float a0 = as[o + ly];
                float a1 = as[o + ly + 16];
                float a2 = as[o + ly + 32];
                float a3 = as[o + ly + 48];
                float b0 = ws[o + lx];
                float b1 = ws[o + lx + 16];
                float b2 = ws[o + lx + 32];
                float b3 = ws[o + lx + 48];
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
        blockedStore(c, cOff, w, bOff, r0, col0, m, n, flags, c00);
        blockedStore(c, cOff, w, bOff, r0, col0 + 16, m, n, flags, c01);
        blockedStore(c, cOff, w, bOff, r0, col0 + 32, m, n, flags, c02);
        blockedStore(c, cOff, w, bOff, r0, col0 + 48, m, n, flags, c03);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0, m, n, flags, c10);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0 + 16, m, n, flags, c11);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0 + 32, m, n, flags, c12);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0 + 48, m, n, flags, c13);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0, m, n, flags, c20);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0 + 16, m, n, flags, c21);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0 + 32, m, n, flags, c22);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0 + 48, m, n, flags, c23);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0, m, n, flags, c30);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0 + 16, m, n, flags, c31);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0 + 32, m, n, flags, c32);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0 + 48, m, n, flags, c33);
    }

    /** Output tile of {@link #gemmBlockedWide}: a 16 × 16 work-group computes 64 rows × 128 columns, 4 × 8 per thread. */
    public static final int WIDE_ROWS = 64;
    public static final int WIDE_COLS = 128;
    /**
     * {@link #gemmBlocked} with a wider tile, for GEMMs with enough rows to fill the GPU with it:
     * {@code C[m, n] (+)= act(A[m, :] · W[n, :] + b[n])} with {@code A} at {@code aOff}, {@code W} ({@code [N, K]}) at {@code wOff} and {@code b}
     * at {@code bOff} in the packed weights, and {@code C} at {@code cOff}. Each work-group stages a
     * 64 × 16 slice of A and a 128 × 16 slice of W in local memory (K-major with a padded pitch, so
     * reads and stores are free of bank conflicts), and each thread accumulates a 4 × 8 block whose
     * rows and columns are 16 apart, so the stores of a warp are contiguous. Flags and epilogue
     * order as {@link #gemm}. Global work {@code (⌈N/128⌉·16, ⌈M/64⌉·16)}, local {@code (16, 16)}.
     */
    public static void gemmBlockedWide(KernelContext ctx, FloatArray a, int aOff, FloatArray w, int wOff, int bOff, FloatArray c,
            int cOff, int m, int n, int k, int flags) {
        int lx = ctx.localIdx;
        int ly = ctx.localIdy;
        int tid = ly * HGEMM_THREADS + lx;
        int rowBase = ctx.groupIdy * WIDE_ROWS;
        int colBase = ctx.groupIdx * WIDE_COLS;
        float[] as = ctx.allocateFloatLocalArray((WIDE_ROWS + 1) * HGEMM_K);
        float[] ws = ctx.allocateFloatLocalArray((WIDE_COLS + 1) * HGEMM_K);
        float c00 = 0f, c01 = 0f, c02 = 0f, c03 = 0f, c04 = 0f, c05 = 0f, c06 = 0f, c07 = 0f;
        float c10 = 0f, c11 = 0f, c12 = 0f, c13 = 0f, c14 = 0f, c15 = 0f, c16 = 0f, c17 = 0f;
        float c20 = 0f, c21 = 0f, c22 = 0f, c23 = 0f, c24 = 0f, c25 = 0f, c26 = 0f, c27 = 0f;
        float c30 = 0f, c31 = 0f, c32 = 0f, c33 = 0f, c34 = 0f, c35 = 0f, c36 = 0f, c37 = 0f;
        for (int kt = 0; kt < k; kt += HGEMM_K) {
            // 64 × 16 elements of A (4 per thread) and 128 × 16 of W (8 per thread); consecutive threads read along K
            for (int e = 0; e < 4; e++) {
                int idx = tid + e * HGEMM_THREADS * HGEMM_THREADS;
                int r = idx / HGEMM_K;
                int kk0 = idx % HGEMM_K;
                int ka = kt + kk0;
                float av = 0f;
                if (rowBase + r < m && ka < k) {
                    av = a.get(aOff + (rowBase + r) * k + ka);
                }
                as[kk0 * (WIDE_ROWS + 1) + r] = av;
            }
            for (int e = 0; e < 8; e++) {
                int idx = tid + e * HGEMM_THREADS * HGEMM_THREADS;
                int r = idx / HGEMM_K;
                int kk0 = idx % HGEMM_K;
                int ka = kt + kk0;
                float wv = 0f;
                if (colBase + r < n && ka < k) {
                    wv = w.get(wOff + (colBase + r) * k + ka);
                }
                ws[kk0 * (WIDE_COLS + 1) + r] = wv;
            }
            ctx.localBarrier();
            for (int kk = 0; kk < HGEMM_K; kk++) {
                int oa = kk * (WIDE_ROWS + 1);
                int ow = kk * (WIDE_COLS + 1);
                float a0 = as[oa + ly];
                float a1 = as[oa + ly + 16];
                float a2 = as[oa + ly + 32];
                float a3 = as[oa + ly + 48];
                float b0 = ws[ow + lx];
                float b1 = ws[ow + lx + 16];
                float b2 = ws[ow + lx + 32];
                float b3 = ws[ow + lx + 48];
                float b4 = ws[ow + lx + 64];
                float b5 = ws[ow + lx + 80];
                float b6 = ws[ow + lx + 96];
                float b7 = ws[ow + lx + 112];
                c00 += a0 * b0;
                c01 += a0 * b1;
                c02 += a0 * b2;
                c03 += a0 * b3;
                c04 += a0 * b4;
                c05 += a0 * b5;
                c06 += a0 * b6;
                c07 += a0 * b7;
                c10 += a1 * b0;
                c11 += a1 * b1;
                c12 += a1 * b2;
                c13 += a1 * b3;
                c14 += a1 * b4;
                c15 += a1 * b5;
                c16 += a1 * b6;
                c17 += a1 * b7;
                c20 += a2 * b0;
                c21 += a2 * b1;
                c22 += a2 * b2;
                c23 += a2 * b3;
                c24 += a2 * b4;
                c25 += a2 * b5;
                c26 += a2 * b6;
                c27 += a2 * b7;
                c30 += a3 * b0;
                c31 += a3 * b1;
                c32 += a3 * b2;
                c33 += a3 * b3;
                c34 += a3 * b4;
                c35 += a3 * b5;
                c36 += a3 * b6;
                c37 += a3 * b7;
            }
            ctx.localBarrier();
        }
        int r0 = rowBase + ly;
        int col0 = colBase + lx;
        blockedStore(c, cOff, w, bOff, r0, col0, m, n, flags, c00);
        blockedStore(c, cOff, w, bOff, r0, col0 + 16, m, n, flags, c01);
        blockedStore(c, cOff, w, bOff, r0, col0 + 32, m, n, flags, c02);
        blockedStore(c, cOff, w, bOff, r0, col0 + 48, m, n, flags, c03);
        blockedStore(c, cOff, w, bOff, r0, col0 + 64, m, n, flags, c04);
        blockedStore(c, cOff, w, bOff, r0, col0 + 80, m, n, flags, c05);
        blockedStore(c, cOff, w, bOff, r0, col0 + 96, m, n, flags, c06);
        blockedStore(c, cOff, w, bOff, r0, col0 + 112, m, n, flags, c07);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0, m, n, flags, c10);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0 + 16, m, n, flags, c11);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0 + 32, m, n, flags, c12);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0 + 48, m, n, flags, c13);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0 + 64, m, n, flags, c14);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0 + 80, m, n, flags, c15);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0 + 96, m, n, flags, c16);
        blockedStore(c, cOff, w, bOff, r0 + 16, col0 + 112, m, n, flags, c17);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0, m, n, flags, c20);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0 + 16, m, n, flags, c21);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0 + 32, m, n, flags, c22);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0 + 48, m, n, flags, c23);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0 + 64, m, n, flags, c24);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0 + 80, m, n, flags, c25);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0 + 96, m, n, flags, c26);
        blockedStore(c, cOff, w, bOff, r0 + 32, col0 + 112, m, n, flags, c27);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0, m, n, flags, c30);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0 + 16, m, n, flags, c31);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0 + 32, m, n, flags, c32);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0 + 48, m, n, flags, c33);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0 + 64, m, n, flags, c34);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0 + 80, m, n, flags, c35);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0 + 96, m, n, flags, c36);
        blockedStore(c, cOff, w, bOff, r0 + 48, col0 + 112, m, n, flags, c37);
    }

    /** One output element of {@link #gemmBlocked} and {@link #gemmBlockedWide}: bias, SiLU or ReLU, accumulate, store. */
    private static void blockedStore(FloatArray c, int cOff, FloatArray w, int bOff, int row, int col, int m, int n, int flags,
            float acc) {
        if (row < m && col < n) {
            float v = acc;
            if ((flags & BIAS) != 0) {
                v += w.get(bOff + col);
            }
            if ((flags & SILU) != 0) {
                v = v / (1f + TornadoMath.exp(-v));
            }
            if ((flags & RELU) != 0) {
                v = TornadoMath.max(v, 0f);
            }
            int ci = cOff + row * n + col;
            if ((flags & ACCUMULATE) != 0) {
                v += c.get(ci);
            }
            c.set(ci, v);
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
    /** {@link #swiGlu} on FP16: {@code out[i] = SiLU(up[r·2f + j]) · up[r·2f + f + j]}, computed in FP32. */
    public static void swiGluHalf(KernelContext ctx, HalfFloatArray up, HalfFloatArray out, int n, int ffn) {
        int id = ctx.globalIdx;
        if (id < n * ffn) {
            int r = id / ffn;
            int j = id - r * ffn;
            float g = up.get(r * 2 * ffn + j).getFloat32();
            float u = up.get(r * 2 * ffn + ffn + j).getFloat32();
            out.set(id, new HalfFloat(g / (1f + TornadoMath.exp(-g)) * u));
        }
    }
    /** {@code acc[i] += x[i]}. */
    public static void addInPlace(KernelContext ctx, FloatArray acc, FloatArray x, int n) {
        int i = ctx.globalIdx;
        if (i < n) {
            acc.set(i, acc.get(i) + x.get(i));
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
            if ((flags & RELU) != 0) {
                v = TornadoMath.max(v, 0f);
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
    /** {@link #biasSilu} with a ReLU: {@code C[i] = max(C[i] + b[col], 0)} (the bias when {@code bOff >= 0}). */
    public static void biasRelu(KernelContext ctx, FloatArray c, FloatArray w, int bOff, int m, int n, HalfFloatArray th,
            int twin) {
        int i = ctx.globalIdx;
        if (i < m * n) {
            int row = i / n;
            int col = i - row * n;
            float v = c.get(i);
            if (bOff >= 0) {
                v += w.get(bOff + col);
            }
            v = TornadoMath.max(v, 0f);
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

    private Kernels() {}
}
