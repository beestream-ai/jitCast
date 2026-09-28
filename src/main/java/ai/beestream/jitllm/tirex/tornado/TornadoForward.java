package ai.beestream.jitllm.tirex.tornado;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.LinkedHashMap;
import java.util.Map;

import ai.beestream.jitllm.tirex.model.ForwardPass;
import ai.beestream.jitllm.tirex.model.ModelBatch;
import ai.beestream.jitllm.tirex.model.TiRex2Config;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * {@code TiRex2.forward()} on a TornadoVM device. One execution plan per batch row count, cached;
 * the packed weights go to the device on each plan's first execution and stay there.
 *
 * <p>P1 scope: batches in which every row is alone in its group (univariate series), so the
 * variate mixers and the reverse cell pass are no-ops. Other batches are rejected.
 */
public final class TornadoForward implements ForwardPass, AutoCloseable {

    private static final String G = "tirex";

    private final TiRex2Config config;
    private final PackedWeights weights;
    private final int maxCachedPlans;
    private final Map<Integer, Plan> plans = new LinkedHashMap<>(16, 0.75f, true);

    public TornadoForward(TiRex2Config config, PackedWeights weights) {
        this(config, weights, 8);
    }

    public TornadoForward(TiRex2Config config, PackedWeights weights, int maxCachedPlans) {
        this.config = config;
        this.weights = weights;
        this.maxCachedPlans = maxCachedPlans;
    }

    @Override
    public float[][][] forward(ModelBatch batch) {
        requireUnivariate(batch);
        int rows = batch.rows();
        Plan plan = plans.get(rows);
        if (plan == null) {
            plan = new Plan(rows);
            plans.put(rows, plan);
            if (plans.size() > maxCachedPlans) {
                Integer eldest = plans.keySet().iterator().next();
                plans.remove(eldest).close();
            }
        }
        return plan.run(batch);
    }

    private static void requireUnivariate(ModelBatch batch) {
        for (int r = 0; r < batch.rows(); r++) {
            if (!batch.target()[r]) {
                throw new UnsupportedOperationException("covariates are not supported by the TornadoVM engine yet (P2)");
            }
            for (int o = r + 1; o < batch.rows(); o++) {
                if (batch.group()[o] == batch.group()[r]) {
                    throw new UnsupportedOperationException("multi-target series are not supported by the TornadoVM engine yet (P2)");
                }
            }
        }
    }

    @Override
    public void close() {
        plans.values().forEach(Plan::close);
        plans.clear();
    }

    /** The buffers and execution plan for one row count. */
    private final class Plan implements AutoCloseable {
        final int rows;
        final FloatArray x, present, tok, stats, h, n, cell, proj, big, y, out;
        final TornadoExecutionPlan exec;

        Plan(int rows) {
            this.rows = rows;
            TiRex2Config c = config;
            int len = c.paddedLength();
            int seq = c.numPatches();
            int d = c.embeddingDim();
            int p = c.patchSize();
            int nq = c.numQuantiles();
            int tokens = rows * seq;
            int ffn = c.ffnDim();
            PackedWeights.Block first = weights.blocks().get(0);
            int projWidth = Math.max(first.mlstmProj() == null ? 0 : first.mlstmProj().out(), ffn);
            int bigWidth = Math.max(Math.max(2 * ffn, 4 * d), Math.max(c.inputFfDim(), c.outputFfDim()));

            x = new FloatArray(rows * len);
            present = new FloatArray(rows * len);
            tok = new FloatArray(tokens * 2 * p);
            stats = new FloatArray(rows * 3);
            h = new FloatArray(tokens * d);
            n = new FloatArray(tokens * d);
            cell = new FloatArray(tokens * d);
            proj = new FloatArray(tokens * projWidth);
            big = new FloatArray(tokens * bigWidth);
            y = new FloatArray(tokens * nq * p);
            out = new FloatArray(rows * nq * len);
            FloatArray w = weights.data();

            KernelContext ctx = new KernelContext();
            GridScheduler grid = new GridScheduler();
            TaskGraph g = new TaskGraph(G)
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, tok, stats, h, n, cell, proj, big, y, out)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, present);
            Builder b = new Builder(g, grid, ctx, w);
            g.task("scale", TiRexKernels::scalePatch, x, present, tok, stats, rows, len, p, c.useArcsinh() ? 1 : 0,
                    c.binaryAware() ? 1 : 0, TiRex2Config.SCALER_EPS);
            b.residual("in", tok, big, h, tokens, weights.inputEmbedding());

            for (int i = 0; i < weights.blocks().size(); i++) {
                PackedWeights.Block blk = weights.blocks().get(i);
                String s = "b" + i + "_";
                b.rms(s + "norm", h, n, blk.normLstm(), tokens, d, TiRex2Config.NORM_EPS);
                if (blk.mlstm()) {
                    PackedWeights.Lin pr = blk.mlstmProj();
                    b.gemm(s + "proj", n, pr, proj, tokens, TiRexKernels.BIAS);
                    int heads = c.timeMixerHeads();
                    int dqk = c.mlstmQkDim() / heads;
                    g.task(s + "mlstm", TiRexKernels::mlstmCell, proj, cell, rows, seq, heads, pr.out(),
                            c.mlstmQkDim(), 2 * c.mlstmQkDim(), 2 * c.mlstmQkDim() + 2 * d,
                            2 * c.mlstmQkDim() + 2 * d + heads, TiRex2Config.GATE_SOFT_CAP, TiRex2Config.MLSTM_EPS);
                    b.headNorm(s + "hnorm", cell, n, blk.headNorm(), tokens, heads, d / heads, proj, pr.out(),
                            2 * c.mlstmQkDim() + d, 1);
                    b.gemm(s + "out", n, blk.outProj(), h, tokens, TiRexKernels.ACCUMULATE);
                    if (dqk != 64 || d / heads != 128) {
                        throw new IllegalStateException("mlstmCell assumes head dims 64/128");
                    }
                } else {
                    int heads = c.slstmHeads();
                    int dh = d / heads;
                    g.task(s + "gates", TiRexKernels::headwiseGates, n, w, blk.slstmGates(), big, tokens, heads, dh);
                    grid.addWorkerGrid(G + "." + s + "slstm", group(rows * heads * dh, dh));
                    g.task(s + "slstm", TiRexKernels::slstmRecurrence, ctx, big, w, blk.recurrent(), blk.slstmBias(),
                            cell, seq, heads);
                    b.headNorm(s + "hnorm", cell, h, blk.groupNorm(), tokens, heads, dh, proj, 0, 0, 2);
                }
                b.rms(s + "fnorm", h, n, blk.normFfn(), tokens, d, TiRex2Config.NORM_EPS);
                b.gemm(s + "up", n, blk.upGateUp(), big, tokens, 0);
                g.task(s + "swiglu", TiRexKernels::swiGlu, big, proj, tokens, blk.ffnDim());
                b.gemm(s + "down", proj, blk.down(), h, tokens, TiRexKernels.ACCUMULATE);
            }
            b.rms("onorm", h, n, weights.stackOutNorm(), tokens, d, c.stackOutNormEps());
            b.residual("out", n, big, y, tokens, weights.outputEmbedding());
            g.task("unpatch", TiRexKernels::unpatch, y, stats, out, rows, seq, nq, p, c.useArcsinh() ? 1 : 0,
                    TiRex2Config.SINH_CLAMP);
            g.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            ImmutableTaskGraph itg = g.snapshot();
            exec = new TornadoExecutionPlan(itg).withGridScheduler(grid);
            String device = System.getProperty("tirex.tornado.device");
            if (device != null && !device.isBlank()) {
                String[] dd = device.split(":");
                exec.withDevice(TornadoExecutionPlan.getDevice(Integer.parseInt(dd[0]), Integer.parseInt(dd[1])));
            }
        }

        float[][][] run(ModelBatch batch) {
            int len = config.paddedLength();
            for (int r = 0; r < rows; r++) {
                float[] row = batch.x()[r];
                int base = r * len;
                for (int t = 0; t < len; t++) {
                    float v = row[t];
                    boolean here = !Float.isNaN(v);
                    x.set(base + t, here ? v : 0f);
                    present.set(base + t, here ? 1f : 0f);
                }
            }
            exec.execute();
            int nq = config.numQuantiles();
            MemorySegment os = out.getSegment();
            float[][][] result = new float[rows][nq][len];
            for (int r = 0; r < rows; r++) {
                for (int q = 0; q < nq; q++) {
                    MemorySegment.copy(os, ValueLayout.JAVA_FLOAT, 4L * (r * nq + q) * len, result[r][q], 0, len);
                }
            }
            return result;
        }

        @Override
        public void close() {
            try {
                exec.close();
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
        }
    }

    private static WorkerGrid group(int global, int local) {
        WorkerGrid wg = new WorkerGrid1D(global);
        wg.setLocalWork(local, 1, 1);
        return wg;
    }

    /** Adds the recurring task shapes (GEMM, norms, residual blocks) with their worker grids. */
    private final class Builder {
        final TaskGraph g;
        final GridScheduler grid;
        final KernelContext ctx;
        final FloatArray w;

        Builder(TaskGraph g, GridScheduler grid, KernelContext ctx, FloatArray w) {
            this.g = g;
            this.grid = grid;
            this.ctx = ctx;
            this.w = w;
        }

        void gemm(String name, FloatArray a, PackedWeights.Lin l, FloatArray c, int m, int flags) {
            int tile = TiRexKernels.TILE;
            WorkerGrid wg = new WorkerGrid2D(ceil(l.out(), tile), ceil(m, tile));
            wg.setLocalWork(tile, tile, 1);
            grid.addWorkerGrid(G + "." + name, wg);
            int f = flags | (l.b() >= 0 ? TiRexKernels.BIAS : 0);
            g.task(name, TiRexKernels::gemm, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, 0, m, l.out(), l.in(), f);
        }

        void rms(String name, FloatArray x, FloatArray y, int wOff, int rows, int d, float eps) {
            grid.addWorkerGrid(G + "." + name, group(rows * TiRexKernels.NORM_THREADS, TiRexKernels.NORM_THREADS));
            g.task(name, TiRexKernels::rmsNorm, ctx, x, y, w, wOff, d, eps);
        }

        void headNorm(String name, FloatArray x, FloatArray y, int wOff, int tokens, int heads, int dh, FloatArray gate,
                int gStride, int gOff, int mode) {
            grid.addWorkerGrid(G + "." + name, group(tokens * heads * dh, dh));
            g.task(name, TiRexKernels::headLayerNorm, ctx, x, y, w, wOff, heads, dh, TiRex2Config.NORM_EPS, gate,
                    gStride, gOff, mode);
        }

        /** {@code y = output(SiLU(hidden(x))) + residual(x)}; {@code scratch} holds the hidden layer. */
        void residual(String name, FloatArray x, FloatArray scratch, FloatArray y, int m, PackedWeights.Residual r) {
            gemm(name + "_hid", x, r.hidden(), scratch, m, TiRexKernels.SILU);
            gemm(name + "_out", scratch, r.output(), y, m, 0);
            gemm(name + "_res", x, r.residual(), y, m, TiRexKernels.ACCUMULATE);
        }

        private int ceil(int v, int t) {
            return (v + t - 1) / t * t;
        }
    }
}
