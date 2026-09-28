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

    private static WorkerGrid grid1(int global, int local) {
        WorkerGrid wg = new WorkerGrid1D(global);
        wg.setLocalWork(local, 1, 1);
        return wg;
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
        run("t", null, g -> g.task("t", TiRexKernels::mlstmCell, fp, fh, rows, SEQ, HEADS, stride, 256, 512, 1536, 1540,
                15f, 1e-6f), fh, fp);
        // the CPU cell from the same projections: identity projections over the fused layout
        float[] exp = Cells.mlstmCore(p, rows, SEQ, HEADS, stride);
        check("mlstmCell", fh.toHeapArray(), exp);
    }

    @Test
    public void slstmGatesAndRecurrence() {
        int rows = 2, n = rows * SEQ, dh = 128;
        float[] x = random(n * D, 1f), wg = random(4 * HEADS * dh * dh, 0.05f);
        FloatArray fx = FloatArray.fromArray(x), fw = FloatArray.fromArray(wg), fo = new FloatArray(n * 4 * D);
        run("t", null, g -> g.task("t", TiRexKernels::headwiseGates, fx, fw, 0, fo, n, HEADS, dh), fo, fx, fw);
        float[][] gates = new float[4][];
        for (int s = 0; s < 4; s++) {
            gates[s] = java.util.Arrays.copyOfRange(wg, s * HEADS * dh * dh, (s + 1) * HEADS * dh * dh);
        }
        float[] expGates = Cells.headwiseGates(x, n, D, HEADS, gates);
        check("headwiseGates", fo.toHeapArray(), expGates);

        float[] r = random(HEADS * dh * 4 * dh, 0.05f), b = random(HEADS * 4 * dh, 0.5f);
        float[] wrb = new float[r.length + b.length];
        System.arraycopy(r, 0, wrb, 0, r.length);
        System.arraycopy(b, 0, wrb, r.length, b.length);
        FloatArray fwx = FloatArray.fromArray(expGates), fwrb = FloatArray.fromArray(wrb), fy = new FloatArray(n * D);
        KernelContext ctx = new KernelContext();
        run("t", grid1(rows * HEADS * dh, dh), g -> g.task("t", TiRexKernels::slstmRecurrence, ctx, fwx, fwrb, 0, r.length,
                fy, SEQ, HEADS), fy, fwx, fwrb);
        check("slstmRecurrence", fy.toHeapArray(), Cells.slstmRecurrence(expGates, rows, SEQ, D, HEADS, r, b));
    }

    @Test
    public void swiGluAndPointwise() {
        int n = 30, ffn = 1408;
        float[] big = random(n * 2 * ffn, 2f);
        FloatArray fb = FloatArray.fromArray(big), fo = new FloatArray(n * ffn);
        run("t", null, g -> g.task("t", TiRexKernels::swiGlu, fb, fo, n, ffn), fo, fb);
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
        run("t", null, g -> g.task("t", TiRexKernels::scalePatch, fx, fp, ft, fs, rows, len, p, 1, 1, 1e-8f), ft, fx, fp, fs);
        float[][] rowsX = new float[rows][];
        for (int r = 0; r < rows; r++) {
            rowsX[r] = java.util.Arrays.copyOfRange(x, r * len, (r + 1) * len);
        }
        check("scalePatch", ft.toHeapArray(), InputStage.apply(rowsX, testConfig()).tokens());
    }

    @Test
    public void variateAttentionWithGroupsAndMask() {
        // two groups: rows 0..2 (target, cov, cov) and rows 3..6 (target, target, cov, cov)
        int nk = 7, dh = 128, stride = 3 * D;
        int[] start = {0, 0, 0, 3, 3, 3, 3}, end = {3, 3, 3, 7, 7, 7, 7}, target = {1, 0, 0, 1, 1, 0, 0};
        float[] p = random(nk * SEQ * stride, 1f);
        FloatArray fp = FloatArray.fromArray(p), fo = new FloatArray(nk * SEQ * D);
        IntArray fs = IntArray.fromArray(start), fe = IntArray.fromArray(end), ft = IntArray.fromArray(target);
        TaskGraph g = new TaskGraph("k").transferToDevice(DataTransferMode.EVERY_EXECUTION, fp, fs, fe, ft, fo)
                .task("t", TiRexKernels::variateAttention, fp, fo, fs, fe, ft, nk, SEQ, HEADS, dh, stride)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, fo);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
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
