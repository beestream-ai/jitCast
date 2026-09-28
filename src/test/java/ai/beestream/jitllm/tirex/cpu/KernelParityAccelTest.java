package ai.beestream.jitllm.tirex.cpu;

import static org.junit.Assert.assertTrue;

import java.util.Random;

import ai.beestream.jitllm.tirex.tornado.TiRexKernels;
import org.junit.Test;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Each TornadoVM kernel on its own, on random inputs, against the CPU reference operator it
 * replaces. Pick the device with {@code -Dtirex.tornado.device=<driver>:<device>}.
 */
public class KernelParityAccelTest {

    private static final double TOLERANCE = 1e-4;
    private static final int SEQ = 74, D = 512, HEADS = 4;
    private final Random rnd = new Random(7);

    private float[] random(int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) rnd.nextGaussian() * scale;
        }
        return v;
    }

    /** Runs one task graph that reads {@code inputs} and returns {@code output}. */
    private static void run(String task, WorkerGrid wg, java.util.function.Consumer<TaskGraph> addTask, FloatArray output,
            FloatArray... inputs) {
        TaskGraph g = new TaskGraph("k").transferToDevice(DataTransferMode.EVERY_EXECUTION, (Object[]) inputs);
        g.transferToDevice(DataTransferMode.EVERY_EXECUTION, output);
        addTask.accept(g);
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, output);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            if (wg != null) {
                GridScheduler grid = new GridScheduler();
                grid.addWorkerGrid("k." + task, wg);
                plan.withGridScheduler(grid);
            }
            String device = System.getProperty("tirex.tornado.device");
            if (device != null && !device.isBlank()) {
                String[] dd = device.split(":");
                plan.withDevice(TornadoExecutionPlan.getDevice(Integer.parseInt(dd[0]), Integer.parseInt(dd[1])));
            }
            plan.execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    /** One thread per item, 128 per group (the engine's element-wise launch). */
    private static WorkerGrid items(int count) {
        return grid1((count + 127) / 128 * 128, 128);
    }

    private static WorkerGrid grid1(int global, int local) {
        WorkerGrid wg = new WorkerGrid1D(global);
        wg.setLocalWork(local, 1, 1);
        return wg;
    }

    private static long maxWorkGroup() {
        String device = System.getProperty("tirex.tornado.device");
        uk.ac.manchester.tornado.api.common.TornadoDevice dev = device != null && !device.isBlank()
                ? TornadoExecutionPlan.getDevice(Integer.parseInt(device.split(":")[0]), Integer.parseInt(device.split(":")[1]))
                : uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        return dev.getDeviceMaxWorkgroupDimensions()[0];
    }

    private static void check(String what, float[] actual, float[] expected) {
        double diff = 0, scale = 1e-6;
        int nan = 0;
        for (int i = 0; i < expected.length; i++) {
            if (Float.isNaN(actual[i]) && !Float.isNaN(expected[i])) {
                nan++;
            }
            diff = Math.max(diff, Math.abs(actual[i] - expected[i]));
            scale = Math.max(scale, Math.abs(expected[i]));
        }
        double err = diff / scale;
        System.out.printf("%-16s rel max err %.2e  NaN %d%n", what, err, nan);
        assertTrue(what + ": " + err + ", NaN " + nan, nan == 0 && err < TOLERANCE);
    }

    @Test
    public void gemmWithEveryFlag() {
        int m = 37, n = 45, k = 70;
        float[] a = random(m * k, 1f), w = random(n * k + n, 0.2f), c0 = random(m * n, 1f);
        for (int flags : new int[] {0, TiRexKernels.BIAS | TiRexKernels.SILU, TiRexKernels.BIAS | TiRexKernels.ACCUMULATE}) {
            FloatArray fa = FloatArray.fromArray(a), fw = FloatArray.fromArray(w), fc = FloatArray.fromArray(c0);
            WorkerGrid wg = new WorkerGrid2D((n + 15) / 16 * 16, (m + 15) / 16 * 16);
            wg.setLocalWork(16, 16, 1);
            KernelContext ctx = new KernelContext();
            int f = flags;
            run("t", wg, g -> g.task("t", TiRexKernels::gemm, ctx, fa, 0, fw, 0, n * k, fc, 0, m, n, k, f), fc, fa, fw);
            float[] bias = java.util.Arrays.copyOfRange(w, n * k, n * k + n);
            float[] exp = Ops.linear(a, m, java.util.Arrays.copyOf(w, n * k), (f & 1) != 0 ? bias : null, k, n);
            if ((f & TiRexKernels.SILU) != 0) {
                Ops.siluInPlace(exp);
            }
            if ((f & TiRexKernels.ACCUMULATE) != 0) {
                Ops.addInPlace(exp, c0);
            }
            check("gemm flags=" + f, fc.toHeapArray(), exp);
        }
    }

    @Test
    public void rmsNormAndHeadNorm() {
        int rows = 9;
        float[] x = random(rows * D, 2f), w = random(D, 1f), gate = random(rows * D, 1f), y0 = random(rows * D, 1f);
        FloatArray fx = FloatArray.fromArray(x), fw = FloatArray.fromArray(w), fy = new FloatArray(rows * D);
        KernelContext ctx = new KernelContext();
        run("t", grid1(rows * 256, 256), g -> g.task("t", TiRexKernels::rmsNorm, ctx, fx, fy, fw, 0, D, 1e-6f), fy, fx, fw);
        check("rmsNorm", fy.toHeapArray(), Ops.rmsNorm(x, rows, D, w, 1e-6f));

        float[] ln = Ops.headLayerNorm(x, rows, HEADS, 128, w, 1e-6f);
        for (int mode = 0; mode < 3; mode++) {
            FloatArray fg = FloatArray.fromArray(gate), fo = FloatArray.fromArray(y0);
            int md = mode;
            run("t", grid1(rows * D, 128), g -> g.task("t", TiRexKernels::headLayerNorm, ctx, fx, fo, fw, 0, HEADS, 128,
                    1e-6f, fg, D, 0, md), fo, fx, fw, fg);
            float[] exp = ln.clone();
            for (int i = 0; i < exp.length; i++) {
                exp[i] = mode == 1 ? (float) (Ops.sigmoid(gate[i]) * exp[i]) : mode == 2 ? exp[i] + y0[i] : exp[i];
            }
            check("headNorm mode=" + mode, fo.toHeapArray(), exp);
        }
    }

    @Test
    public void mlstmCell() {
        int rows = 2, stride = 1544;
        float[] p = random(rows * SEQ * stride, 1f);
        FloatArray fp = FloatArray.fromArray(p), fh = new FloatArray(rows * SEQ * D);
        KernelContext ctx = new KernelContext();
        run("t", grid1(rows * HEADS * SEQ * 128, 128), g -> g.task("t", TiRexKernels::mlstmCell, ctx, fp, fh, rows, SEQ, HEADS, stride,
                256, 512, 1536, 1540, 15f, 1e-6f), fh, fp);
        // the CPU cell from the same projections: identity projections over the fused layout
        float[] exp = Cells.mlstmCore(p, rows, SEQ, HEADS, stride);
        check("mlstmCell", fh.toHeapArray(), exp);
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
        run("t", wgrid, g -> g.task("t", TiRexKernels::gemm, gctx, fx, 0, fw, 0, 0, fo, 0, n, 4 * D, D, 0), fo, fx, fw);
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
        }
    }

    @Test
    public void swiGluAndPointwise() {
        int n = 30, ffn = 1408;
        float[] big = random(n * 2 * ffn, 2f);
        FloatArray fb = FloatArray.fromArray(big), fo = new FloatArray(n * ffn);
        KernelContext ctx = new KernelContext();
        run("t", items(n * ffn), g -> g.task("t", TiRexKernels::swiGlu, ctx, fb, fo, n, ffn), fo, fb);
        float[] exp = new float[n * ffn];
        for (int r = 0; r < n; r++) {
            for (int j = 0; j < ffn; j++) {
                exp[r * ffn + j] = Ops.silu(big[r * 2 * ffn + j]) * big[r * 2 * ffn + ffn + j];
            }
        }
        check("swiGlu", fo.toHeapArray(), exp);
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
        run("t", grid1(rows * 256, 256), g -> g.task("t", TiRexKernels::scalePatch, ctx, fx, fp, ft, fs, rows, len, p, 1, 1, 1e-8f), ft,
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
                .task("t", TiRexKernels::variateAttention, new KernelContext(), fp, fo, fs, fe, ft, nk, SEQ, HEADS, dh, stride)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, fo);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            GridScheduler grid = new GridScheduler();
            grid.addWorkerGrid("k.t", grid1((nk + TiRexKernels.ATTN_QUERIES - 1) / TiRexKernels.ATTN_QUERIES * SEQ * HEADS * 128, 128));
            plan.withGridScheduler(grid);
            String device = System.getProperty("tirex.tornado.device");
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

    private static ai.beestream.jitllm.tirex.model.TiRex2Config testConfig() {
        return new ai.beestream.jitllm.tirex.model.TiRex2Config(12, 512, 32, 2048, 2048, 2048, 320, new float[9], true,
                true, 1e-7f, true, false, java.util.List.of(), 4, 4, 4);
    }
}
