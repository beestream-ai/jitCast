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
package ai.beestream.jitrex.tornado;

import static ai.beestream.jitrex.tornado.KernelHarness.NO_TWIN;
import static ai.beestream.jitrex.tornado.KernelHarness.check;
import static ai.beestream.jitrex.tornado.KernelHarness.execute;
import static ai.beestream.jitrex.tornado.KernelHarness.fromTwin;
import static ai.beestream.jitrex.tornado.KernelHarness.grid1;
import static ai.beestream.jitrex.tornado.KernelHarness.half;
import static ai.beestream.jitrex.tornado.KernelHarness.items;
import static ai.beestream.jitrex.tornado.KernelHarness.roundHalf;
import static ai.beestream.jitrex.tornado.KernelHarness.run;

import java.util.Arrays;
import java.util.Random;

import ai.beestream.jitrex.cpu.Ops;
import org.junit.Assume;
import org.junit.Test;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoDeviceTileNotSupported;
import uk.ac.manchester.tornado.api.tile.TileContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Each shared kernel ({@link Kernels}) on its own, on random inputs, against the CPU operator it
 * replaces ({@link Ops}). Pick the device with {@code -Dtirex.tornado.device=<driver>:<device>}.
 */
public class KernelsAccelTest {

    private static final int D = 512;
    private final Random rnd = new Random(7);

    private float[] random(int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) rnd.nextGaussian() * scale;
        }
        return v;
    }

    @Test
    public void gemmWithEveryFlag() {
        int m = 37, n = 45, k = 70;
        float[] a = random(m * k, 1f), w = random(n * k + n, 0.2f), c0 = random(m * n, 1f);
        for (int flags : new int[] {0, Kernels.BIAS | Kernels.SILU, Kernels.BIAS | Kernels.ACCUMULATE}) {
            FloatArray fa = FloatArray.fromArray(a), fw = FloatArray.fromArray(w), fc = FloatArray.fromArray(c0);
            WorkerGrid wg = new WorkerGrid2D((n + 15) / 16 * 16, (m + 15) / 16 * 16);
            wg.setLocalWork(16, 16, 1);
            KernelContext ctx = new KernelContext();
            int f = flags;
            run("t", wg, g -> g.task("t", Kernels::gemm, ctx, fa, 0, fw, 0, n * k, fc, 0, m, n, k, f), fc, fa, fw);
            float[] bias = Arrays.copyOfRange(w, n * k, n * k + n);
            float[] exp = Ops.linear(a, m, Arrays.copyOf(w, n * k), (f & Kernels.BIAS) != 0 ? bias : null, k, n);
            if ((f & Kernels.SILU) != 0) {
                Ops.siluInPlace(exp);
            }
            if ((f & Kernels.ACCUMULATE) != 0) {
                Ops.addInPlace(exp, c0);
            }
            check("gemm flags=" + f, fc.toHeapArray(), exp);
        }
    }

    /** The FP16 GEMMs: FP16-rounded A and W, FP32 bias, accumulation and C, plus C's FP16 twin. */
    private void checkHalfGemm(boolean tile) {
        int m = 150, n = 200, k = 88; // ragged in every dimension against 64 × 64 × 16/32 tiles
        float[] a = roundHalf(random(m * k, 1f)), wm = roundHalf(random(n * k, 0.2f)), b = random(n, 0.5f), c0 = random(m * n, 1f);
        int[] flagSets = {0, Kernels.BIAS | Kernels.SILU, Kernels.BIAS | Kernels.ACCUMULATE, Kernels.ACCUMULATE};
        for (int flags : flagSets) {
            HalfFloatArray ha = half(a), hw = half(wm);
            HalfFloatArray th = new HalfFloatArray(m * n);
            FloatArray fc = FloatArray.fromArray(c0), fb = FloatArray.fromArray(b);
            TaskGraph g = new TaskGraph("k").transferToDevice(DataTransferMode.EVERY_EXECUTION, ha, hw, fc, fb, th);
            WorkerGrid wg;
            boolean bias = (flags & Kernels.BIAS) != 0;
            if (tile) {
                wg = new WorkerGrid2D((m + Kernels.CT_M - 1) / Kernels.CT_M, (n + Kernels.CT_N - 1) / Kernels.CT_N);
                float fb1 = bias ? 1f : 0f, fs = (flags & Kernels.SILU) != 0 ? 1f : 0f, fa = (flags & Kernels.ACCUMULATE) != 0 ? 1f : 0f;
                g.task("t", Kernels::gemmHalfTile, new TileContext(), ha, hw, fc, fb, 0, m, n, k, fb1, fs, fa, th, 1);
            } else {
                wg = new WorkerGrid2D((n + 63) / 64 * 16, (m + 63) / 64 * 16);
                wg.setLocalWork(16, 16, 1);
                KernelContext ctx = new KernelContext();
                g.task("t", Kernels::gemmHalf, ctx, ha, hw, fc, fb, 0, m, n, k, flags, th, 1);
            }
            g.transferToHost(DataTransferMode.EVERY_EXECUTION, fc, th);
            execute(g, wg);
            float[] exp = Ops.linear(a, m, wm, bias ? b : null, k, n);
            if ((flags & Kernels.SILU) != 0) {
                Ops.siluInPlace(exp);
            }
            if ((flags & Kernels.ACCUMULATE) != 0) {
                Ops.addInPlace(exp, c0);
            }
            String what = (tile ? "gemmHalfTile" : "gemmHalf") + " flags=" + flags;
            check(what, fc.toHeapArray(), exp);
            check(what + " twin", fromTwin(th, m * n), roundHalf(fc.toHeapArray()));
        }
    }

    @Test
    public void gemmHalfKernelContext() {
        checkHalfGemm(false);
    }

    @Test
    public void gemmHalfTileApi() {
        Assume.assumeTrue("CUDA Tile needs the CUDA backend", KernelHarness.device().getTornadoVMBackend() == TornadoVMBackendType.CUDA);
        try {
            checkHalfGemm(true);
        } catch (IllegalStateException e) {
            // no CUDA Tile toolchain here (toolkit 13.3+, or -Dtornado.cuda.nvcc): skip rather than fail
            for (Throwable c = e; c != null; c = c.getCause()) {
                Assume.assumeFalse(String.valueOf(c.getMessage()), c instanceof TornadoDeviceTileNotSupported);
            }
            throw e;
        }
    }

    @Test
    public void rmsNorm() {
        int rows = 9;
        float[] x = random(rows * D, 2f), w = random(D, 1f);
        FloatArray fx = FloatArray.fromArray(x), fw = FloatArray.fromArray(w), fy = new FloatArray(rows * D);
        KernelContext ctx = new KernelContext();
        run("t", grid1(rows * Kernels.NORM_THREADS, Kernels.NORM_THREADS),
                g -> g.task("t", Kernels::rmsNorm, ctx, fx, fy, fw, 0, D, 1e-6f, NO_TWIN, 0), fy, fx, fw);
        check("rmsNorm", fy.toHeapArray(), Ops.rmsNorm(x, rows, D, w, 1e-6f));
    }

    @Test
    public void rmsNormWritesTheFp16Twin() {
        int rows = 5;
        float[] x = random(rows * D, 2f), w = random(D, 1f);
        FloatArray fx = FloatArray.fromArray(x), fw = FloatArray.fromArray(w), fy = new FloatArray(rows * D);
        HalfFloatArray fh = new HalfFloatArray(rows * D);
        KernelContext ctx = new KernelContext();
        TaskGraph g = new TaskGraph("k").transferToDevice(DataTransferMode.EVERY_EXECUTION, fx, fw, fy, fh)
                .task("t", Kernels::rmsNorm, ctx, fx, fy, fw, 0, D, 1e-6f, fh, 1)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, fy, fh);
        execute(g, grid1(rows * Kernels.NORM_THREADS, Kernels.NORM_THREADS));
        check("rmsNorm FP16 twin", fromTwin(fh, rows * D), roundHalf(fy.toHeapArray()));
    }

    @Test
    public void swiGlu() {
        int n = 30, ffn = 1408;
        float[] big = random(n * 2 * ffn, 2f);
        FloatArray fb = FloatArray.fromArray(big), fo = new FloatArray(n * ffn);
        KernelContext ctx = new KernelContext();
        run("t", items(n * ffn), g -> g.task("t", Kernels::swiGlu, ctx, fb, fo, n, ffn, NO_TWIN, 0), fo, fb);
        float[] exp = new float[n * ffn];
        for (int r = 0; r < n; r++) {
            for (int j = 0; j < ffn; j++) {
                exp[r * ffn + j] = Ops.silu(big[r * 2 * ffn + j]) * big[r * 2 * ffn + ffn + j];
            }
        }
        check("swiGlu", fo.toHeapArray(), exp);
    }

    @Test
    public void biasSiluAndAddRmsNorm() {
        // biasSilu, with and without a bias
        int m = 29, n = 70;
        float[] c0 = random(m * n, 2f), wb = random(n, 1f);
        for (int withBias : new int[] {1, 0}) {
            FloatArray fc = FloatArray.fromArray(c0), fw = FloatArray.fromArray(wb);
            HalfFloatArray th = new HalfFloatArray(m * n);
            KernelContext ctx = new KernelContext();
            int bOff = withBias == 1 ? 0 : -1;
            run("t", items(m * n), g -> g.task("t", Kernels::biasSilu, ctx, fc, fw, bOff, m, n, th, 1), fc, fw);
            float[] exp = c0.clone();
            for (int i = 0; i < exp.length; i++) {
                exp[i] = Ops.silu(exp[i] + (withBias == 1 ? wb[i % n] : 0f));
            }
            check("biasSilu bias=" + withBias, fc.toHeapArray(), exp);
        }

        // addRmsNorm: h += y, n = RMSNorm(h)·w
        int rows = 7;
        float[] h0 = random(rows * D, 1f), y0 = random(rows * D, 1f), w = random(D, 1f);
        FloatArray fh = FloatArray.fromArray(h0), fy = FloatArray.fromArray(y0), fn = new FloatArray(rows * D), fw = FloatArray.fromArray(w);
        HalfFloatArray tn = new HalfFloatArray(rows * D);
        KernelContext ctx = new KernelContext();
        TaskGraph g = new TaskGraph("k").transferToDevice(DataTransferMode.EVERY_EXECUTION, fh, fy, fn, fw, tn)
                .task("t", Kernels::addRmsNorm, ctx, fh, fy, fn, fw, 0, D, 1e-6f, tn, 1)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, fh, fn, tn);
        execute(g, grid1(rows * Kernels.NORM_THREADS, Kernels.NORM_THREADS));
        float[] hExp = h0.clone();
        Ops.addInPlace(hExp, y0);
        float[] nExp = Ops.rmsNorm(hExp, rows, D, w, 1e-6f);
        check("addRmsNorm h", fh.toHeapArray(), hExp);
        check("addRmsNorm n", fn.toHeapArray(), nExp);
        check("addRmsNorm twin", fromTwin(tn, rows * D), roundHalf(fn.toHeapArray()));
    }
}
