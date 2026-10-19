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
package ai.beestream.jitcast.tirex2.cpu;

import static ai.beestream.jitcast.tornado.KernelHarness.NO_TWIN;
import static ai.beestream.jitcast.tornado.KernelHarness.check;
import static ai.beestream.jitcast.tornado.KernelHarness.execute;
import static ai.beestream.jitcast.tornado.KernelHarness.fromTwin;
import static ai.beestream.jitcast.tornado.KernelHarness.grid1;
import static ai.beestream.jitcast.tornado.KernelHarness.maxWorkGroup;
import static ai.beestream.jitcast.tornado.KernelHarness.roundHalf;
import static ai.beestream.jitcast.tornado.KernelHarness.run;

import java.util.Random;

import ai.beestream.jitcast.cpu.Ops;
import ai.beestream.jitcast.tirex2.tornado.TiRexKernels;
import ai.beestream.jitcast.tornado.Kernels;
import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Each TiRex-2-specific TornadoVM kernel ({@link TiRexKernels}) on its own, on random inputs,
 * against the CPU reference operator it replaces. The shared kernels are tested in
 * {@code tornado.KernelsAccelTest}. Pick the device with
 * {@code -Djitcast.tornado.device=<driver>:<device>}.
 */
public class KernelParityAccelTest {

    private static final int SEQ = 74, D = 512, HEADS = 4;
    private final Random rnd = new Random(7);

    private float[] random(int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) rnd.nextGaussian() * scale;
        }
        return v;
    }

    @Test
    public void headNorm() {
        int rows = 9;
        float[] x = random(rows * D, 2f), w = random(D, 1f), gate = random(rows * D, 1f), y0 = random(rows * D, 1f);
        FloatArray fx = FloatArray.fromArray(x), fw = FloatArray.fromArray(w);
        KernelContext ctx = new KernelContext();

        float[] ln = TiRexOps.headLayerNorm(x, rows, HEADS, 128, w, 1e-6f);
        for (int mode = 0; mode < 3; mode++) {
            FloatArray fg = FloatArray.fromArray(gate), fo = FloatArray.fromArray(y0);
            int md = mode;
            run("t", grid1(rows * D, 128), g -> g.task("t", TiRexKernels::headLayerNorm, ctx, fx, fo, fw, 0, HEADS, 128,
                    1e-6f, fg, D, 0, md, NO_TWIN, 0), fo, fx, fw, fg);
            float[] exp = ln.clone();
            for (int i = 0; i < exp.length; i++) {
                exp[i] = mode == 1 ? (float) (Ops.sigmoid(gate[i]) * exp[i]) : mode == 2 ? exp[i] + y0[i] : exp[i];
            }
            check("headNorm mode=" + mode, fo.toHeapArray(), exp);
        }
    }

    @Test
    public void mlstmCell() {
        // 74 patches (the base model) and 285 (the GIFT-Eval checkpoints: steps beyond one per thread)
        for (int seq : new int[] {SEQ, 285}) {
            int rows = 2, stride = 1544;
            float[] p = random(rows * seq * stride, 1f);
            FloatArray fp = FloatArray.fromArray(p), fh = new FloatArray(rows * seq * D);
            KernelContext ctx = new KernelContext();
            run("t", grid1(rows * HEADS * seq * 128, 128), g -> g.task("t", TiRexKernels::mlstmCell, ctx, fp, fh, rows, seq, HEADS,
                    stride, 256, 512, 1536, 1540, 15f, 1e-6f), fh, fp);
            // the CPU cell from the same projections: identity projections over the fused layout
            check("mlstmCell seq=" + seq, fh.toHeapArray(), Cells.mlstmCore(p, rows, seq, HEADS, stride));
        }
    }

    @Test
    public void mlstmChunk() {
        // 74 and 285 patches, plus lengths that leave a partial last block (1, 17, 31)
        for (int seq : new int[] {SEQ, 285, 1, 17, 31}) {
            int rows = 2, stride = 1544;
            float[] p = random(rows * seq * stride, 1f);
            FloatArray fp = FloatArray.fromArray(p), fh = new FloatArray(rows * seq * D);
            KernelContext ctx = new KernelContext();
            int blocks = (seq + TiRexKernels.MLSTM_TB - 1) / TiRexKernels.MLSTM_TB;
            run("t", grid1(rows * HEADS * blocks * 128, 128), g -> g.task("t", TiRexKernels::mlstmChunk, ctx, fp, fh, rows, seq, HEADS,
                    stride, 256, 512, 1536, 1540, 15f, 1e-6f), fh, fp);
            check("mlstmChunk seq=" + seq, fh.toHeapArray(), Cells.mlstmCore(p, rows, seq, HEADS, stride));
        }
    }

    @Test
    public void slstmGatesAndRecurrence() {
        int rows = 3, n = rows * SEQ, dh = 128;
        float[] x = random(n * D, 1f), wg = random(4 * HEADS * dh * dh, 0.05f);
        float[][] gates = new float[4][];
        for (int s = 0; s < 4; s++) {
            gates[s] = java.util.Arrays.copyOfRange(wg, s * HEADS * dh * dh, (s + 1) * HEADS * dh * dh);
        }
        float[] expGates = Cells.headwiseGates(x, n, D, HEADS, gates);
        // the engine runs the four headwise projections as one block-diagonal [4D, D] GEMM
        float[] dense = new float[4 * D * D];
        for (int g = 0; g < 4; g++) {
            for (int h = 0; h < HEADS; h++) {
                for (int o = 0; o < dh; o++) {
                    System.arraycopy(gates[g], (h * dh + o) * dh, dense, (g * D + h * dh + o) * D + h * dh, dh);
                }
            }
        }
        FloatArray fx = FloatArray.fromArray(x), fw = FloatArray.fromArray(dense), fo = new FloatArray(n * 4 * D);
        WorkerGrid wgrid = new WorkerGrid2D(4 * D, (n + 15) / 16 * 16);
        wgrid.setLocalWork(16, 16, 1);
        KernelContext gctx = new KernelContext();
        run("t", wgrid, g -> g.task("t", Kernels::gemm, gctx, fx, 0, fw, 0, 0, fo, 0, n, 4 * D, D, 0), fo, fx, fw);
        check("denseGates", fo.toHeapArray(), expGates);

        float[] r = random(HEADS * dh * 4 * dh, 0.05f), b = random(HEADS * 4 * dh, 0.5f);
        float[] wrb = new float[r.length + b.length];
        System.arraycopy(r, 0, wrb, 0, r.length);
        System.arraycopy(b, 0, wrb, r.length, b.length);
        FloatArray fwx = FloatArray.fromArray(expGates), fwrb = FloatArray.fromArray(wrb), fy = new FloatArray(n * D);
        KernelContext ctx = new KernelContext();
        float[] expY = Cells.slstmRecurrence(expGates, rows, SEQ, D, HEADS, r, b);
        for (int gpt : new int[] {1, 2, 4}) {
            int threads = 4 * dh / gpt;
            if (threads > maxWorkGroup()) {
                continue;
            }
            for (int rb : new int[] {1, 2, 8}) {
                FloatArray o = new FloatArray(n * D);
                int groups = (rows + rb - 1) / rb * HEADS;
                run("t", grid1(groups * threads, threads), g -> g.task("t", TiRexKernels::slstmRecurrence, ctx, fwx, fwrb, 0,
                        r.length, o, rows, SEQ, HEADS, gpt, rb), o, fwx, fwrb);
                check("slstmRecurrence gpt=" + gpt + " rows/group=" + rb, o.toHeapArray(), expY);
            }
            // FP16 R: exact against the CPU recurrence on R rounded to FP16
            float[] rRounded = r.clone();
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray rh = new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(r.length);
            for (int i = 0; i < r.length; i++) {
                short bits = Float.floatToFloat16(r[i]);
                rRounded[i] = Float.float16ToFloat(bits);
                rh.getSegment().setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, i, bits);
            }
            float[] expHalf = Cells.slstmRecurrence(expGates, rows, SEQ, D, HEADS, rRounded, b);
            FloatArray oh = new FloatArray(n * D);
            FloatArray bOnly = FloatArray.fromArray(b);
            run("t", grid1((rows + 1) / 2 * HEADS * threads, threads), g -> g.task("t", TiRexKernels::slstmRecurrenceHalfR, ctx, fwx,
                    bOnly, rh, 0, 0, oh, rows, SEQ, HEADS, gpt, 2), oh, fwx, bOnly);
            check("slstmRecurrence FP16 R gpt=" + gpt, oh.toHeapArray(), expHalf);
            // FP16 R, half of it held in registers: 512 threads, so only where the device allows them
            if (gpt == 1) {
                // up to TornadoForward.SLSTM_REG_MAX_ROWS rows per group: 4 or more exceed the register file on CUDA and OpenCL
                for (int rb : new int[] {1, 2}) {
                    FloatArray or = new FloatArray(n * D);
                    int groups = (rows + rb - 1) / rb * HEADS;
                    run("t", grid1(groups * threads, threads), g -> g.task("t", TiRexKernels::slstmRecurrenceRegR, ctx, fwx,
                            bOnly, rh, 0, 0, or, rows, SEQ, HEADS, rb), or, fwx, bOnly);
                    check("slstmRecurrence FP16 R in registers rows/group=" + rb, or.toHeapArray(), expHalf);
                }
            }
        }
    }

    @Test
    public void scalePatch() {
        int rows = 3, len = 2368, p = 32;
        float[] x = random(rows * len, 5f);
        for (int i = 0; i < 700; i++) {
            x[i] = Float.NaN; // left padding on row 0
        }
        for (int i = len; i < 2 * len; i++) {
            x[i] = (i % 3 == 0) ? 1f : 0f; // a binary row
        }
        float[] clean = new float[x.length], here = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            here[i] = Float.isNaN(x[i]) ? 0f : 1f;
            clean[i] = Float.isNaN(x[i]) ? 0f : x[i];
        }
        FloatArray fx = FloatArray.fromArray(clean), fp = FloatArray.fromArray(here);
        FloatArray ft = new FloatArray(rows * 74 * 64), fs = new FloatArray(rows * 3);
        KernelContext ctx = new KernelContext();
        run("t", grid1(rows * 256, 256), g -> g.task("t", TiRexKernels::scalePatch, ctx, fx, fp, ft, fs, rows, len, p, 1, 1, 1e-8f, NO_TWIN, 0), ft,
                fx, fp, fs);
        float[][] rowsX = new float[rows][];
        for (int r = 0; r < rows; r++) {
            rowsX[r] = java.util.Arrays.copyOfRange(x, r * len, (r + 1) * len);
        }
        check("scalePatch", ft.toHeapArray(), InputStage.apply(rowsX, testConfig()).tokens());
    }

    @Test
    public void variateAttentionWithGroupsAndMask() {
        // two groups: rows 0..2 (target, cov, cov) and rows 3..6 (target, target, cov, cov)
        // groups: rows 0..2 (target, cov, cov), 3..6 (target, target, cov, cov), 7..26 (target + 19 covs),
        // so query blocks of 16 straddle groups and the last block is partial
        int nk = 27, dh = 128, stride = 3 * D;
        int[] start = new int[nk], end = new int[nk], target = new int[nk];
        for (int i = 0; i < nk; i++) {
            start[i] = i < 3 ? 0 : i < 7 ? 3 : 7;
            end[i] = i < 3 ? 3 : i < 7 ? 7 : 27;
            target[i] = (i == 0 || i == 3 || i == 4 || i == 7) ? 1 : 0;
        }
        float[] p = random(nk * SEQ * stride, 1f);
        FloatArray fp = FloatArray.fromArray(p), fo = new FloatArray(nk * SEQ * D);
        IntArray fs = IntArray.fromArray(start), fe = IntArray.fromArray(end), ft = IntArray.fromArray(target);
        TaskGraph g = new TaskGraph("k").transferToDevice(DataTransferMode.EVERY_EXECUTION, fp, fs, fe, ft, fo)
                .task("t", TiRexKernels::variateAttention, new KernelContext(), fp, fo, fs, fe, ft, nk, SEQ, HEADS, dh, stride, NO_TWIN, 0)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, fo);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            GridScheduler grid = new GridScheduler();
            grid.addWorkerGrid("k.t", grid1((nk + TiRexKernels.ATTN_QUERIES - 1) / TiRexKernels.ATTN_QUERIES * SEQ * HEADS * 128, 128));
            plan.withGridScheduler(grid);
            String device = System.getProperty("jitcast.tornado.device");
            if (device != null && !device.isBlank()) {
                String[] dd = device.split(":");
                plan.withDevice(TornadoExecutionPlan.getDevice(Integer.parseInt(dd[0]), Integer.parseInt(dd[1])));
            }
            plan.execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        float[] exp = new float[nk * SEQ * D];
        double scale = 1 / Math.sqrt(dh);
        for (int q = 0; q < nk; q++) {
            for (int t = 0; t < SEQ; t++) {
                for (int h = 0; h < HEADS; h++) {
                    int qo = (q * SEQ + t) * stride + h * dh;
                    double[] sc = new double[nk];
                    double max = Double.NEGATIVE_INFINITY;
                    for (int k = start[q]; k < end[q]; k++) {
                        if (target[q] == 1 || target[k] == 0) {
                            double dot = 0;
                            for (int j = 0; j < dh; j++) {
                                dot += (double) p[qo + j] * p[(k * SEQ + t) * stride + D + h * dh + j];
                            }
                            sc[k] = dot * scale;
                            max = Math.max(max, sc[k]);
                        }
                    }
                    double sum = 0;
                    double[] acc = new double[dh];
                    for (int k = start[q]; k < end[q]; k++) {
                        if (target[q] == 1 || target[k] == 0) {
                            double e = Math.exp(sc[k] - max);
                            sum += e;
                            for (int j = 0; j < dh; j++) {
                                acc[j] += e * p[(k * SEQ + t) * stride + 2 * D + h * dh + j];
                            }
                        }
                    }
                    for (int j = 0; j < dh; j++) {
                        exp[(q * SEQ + t) * D + h * dh + j] = (float) (acc[j] / sum);
                    }
                }
            }
        }
        check("variateAttention", fo.toHeapArray(), exp);
    }

    @Test
    public void qkNormPerHead() {
        int tokens = 37, dh = 128, stride = 3 * D;
        float[] p = random(tokens * stride, 3f);
        float[] exp = p.clone();
        for (int tok = 0; tok < tokens; tok++) {
            // q and k (the first 2·D values of the token) per head; v untouched
            float[] qk = java.util.Arrays.copyOfRange(p, tok * stride, tok * stride + 2 * D);
            float[] nrm = Ops.rmsNorm(qk, 2 * HEADS, dh, null, Math.ulp(1.0f));
            System.arraycopy(nrm, 0, exp, tok * stride, 2 * D);
        }
        KernelContext ctx = new KernelContext();
        float eps = Math.ulp(1.0f);
        for (int vpp : new int[] {1, 2, 8}) {
            if ((long) vpp * dh > maxWorkGroup()) {
                continue;
            }
            FloatArray fp = FloatArray.fromArray(p);
            run("t", grid1(tokens * vpp * dh, vpp * dh), g -> g.task("t", TiRexKernels::qkNorm, ctx, fp, tokens, HEADS, dh, stride, eps, vpp), fp, fp);
            check("qkNorm vectors/pass=" + vpp, fp.toHeapArray(), exp);
        }
    }

    @Test
    public void headNormAddRmsNorm() {
        int rows = 7;
        float[] h0 = random(rows * D, 1f);
        KernelContext ctx = new KernelContext();
        // per-head LN of x added into h, then RMSNorm into n
        if (maxWorkGroup() >= TiRexKernels.FUSED_NORM_THREADS) {
            float[] x0 = random(rows * D, 1f), wg = random(D, 1f), wr = random(D, 1f);
            float[] packed = new float[2 * D];
            System.arraycopy(wg, 0, packed, 0, D);
            System.arraycopy(wr, 0, packed, D, D);
            FloatArray fx = FloatArray.fromArray(x0), fh2 = FloatArray.fromArray(h0), fn2 = new FloatArray(rows * D);
            FloatArray fp = FloatArray.fromArray(packed);
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray tn2 = new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(rows * D);
            TaskGraph g2 = new TaskGraph("k").transferToDevice(DataTransferMode.EVERY_EXECUTION, fx, fh2, fn2, fp, tn2)
                    .task("t", TiRexKernels::headNormAddRmsNorm, ctx, fx, fh2, fn2, fp, 0, D, HEADS, 128, 1e-6f, 1e-6f, tn2, 1)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, fh2, fn2, tn2);
            execute(g2, grid1(rows * D, D));
            float[] h2 = h0.clone();
            Ops.addInPlace(h2, TiRexOps.headLayerNorm(x0, rows, HEADS, 128, wg, 1e-6f));
            check("headNormAddRms h", fh2.toHeapArray(), h2);
            check("headNormAddRms n", fn2.toHeapArray(), Ops.rmsNorm(h2, rows, D, wr, 1e-6f));
            check("headNormAddRms twin", fromTwin(tn2, rows * D), roundHalf(fn2.toHeapArray()));
        }
    }

    @Test
    public void slstmRecurrenceLongSequence() {
        int rows = 2, seq = 285, n = rows * seq, dh = 128;
        float[] wx = random(n * 4 * D, 1f), r = random(HEADS * dh * 4 * dh, 0.05f), b = random(HEADS * 4 * dh, 0.5f);
        float[] wrb = new float[r.length + b.length];
        System.arraycopy(r, 0, wrb, 0, r.length);
        System.arraycopy(b, 0, wrb, r.length, b.length);
        FloatArray fwx = FloatArray.fromArray(wx), fwrb = FloatArray.fromArray(wrb), fy = new FloatArray(n * D);
        KernelContext ctx = new KernelContext();
        int threads = (int) Math.min(512, maxWorkGroup());
        int gpt = 512 / threads;
        run("t", grid1(rows * HEADS * threads, threads), g -> g.task("t", TiRexKernels::slstmRecurrence, ctx, fwx, fwrb, 0, r.length,
                fy, rows, seq, HEADS, gpt, 1), fy, fwx, fwrb);
        check("slstmRecurrence seq=285", fy.toHeapArray(), Cells.slstmRecurrence(wx, rows, seq, D, HEADS, r, b));
    }

    private static ai.beestream.jitcast.tirex2.TiRex2Config testConfig() {
        return new ai.beestream.jitcast.tirex2.TiRex2Config(12, 512, 32, 2048, 2048, 2048, 320, new float[9], true,
                true, 1e-7f, true, false, java.util.List.of(), 4, 4, 4, true);
    }
}
