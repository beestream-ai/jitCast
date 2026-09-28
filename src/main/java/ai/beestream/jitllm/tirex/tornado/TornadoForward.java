package ai.beestream.jitllm.tirex.tornado;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * {@code TiRex2.forward()} on a TornadoVM device. A plan is built per batch {@link Shape} (rows,
 * known covariates, rows in multi-row groups) and cached; which rows those are is per-call data,
 * so every batch of the same shape reuses one compiled plan. The packed weights go to the device
 * on each plan's first execution and stay there.
 */
public final class TornadoForward implements ForwardPass, AutoCloseable {

    private static final String G = "tirex";

    /** Work-group size of the element-wise kernels. */
    private static final int ITEM_GROUP = 128;

    /** How GEMMs run: the portable tiled kernel, or cuBLAS (FP32, or TF32 tensor cores) on CUDA. */
    public enum GemmMode {
        TILED, CUBLAS, CUBLAS_TF32;

        static GemmMode fromProperty() {
            String v = System.getProperty("tirex.tornado.gemm", "tiled").trim().toUpperCase().replace('-', '_');
            return GemmMode.valueOf(v);
        }
    }

    private final GemmMode gemmMode = GemmMode.fromProperty();

    /** What a compiled plan depends on: the row count and the sizes of the two gathered subsets. */
    public record Shape(int rows, int known, int kept) {}

    /**
     * The per-call structure of a batch: the known covariates (reverse cell pass) and, for the
     * variate attention, the rows in multi-row groups with each one's group range and target flag.
     */
    record Structure(int[] known, int[] kept, int[] groupStart, int[] groupEnd, int[] keptTarget) {
        Shape shape(int rows) {
            return new Shape(rows, known.length, kept.length);
        }
    }

    private final TiRex2Config config;
    private final PackedWeights weights;
    private final int maxCachedPlans;
    private final Map<Shape, Plan> plans = new LinkedHashMap<>(16, 0.75f, true);

    public TornadoForward(TiRex2Config config, PackedWeights weights) {
        this(config, weights, 8);
    }

    public TornadoForward(TiRex2Config config, PackedWeights weights, int maxCachedPlans) {
        this.config = config;
        this.weights = weights;
        this.maxCachedPlans = maxCachedPlans;
    }

    /**
     * Gates per thread in the sLSTM recurrence: 1 (512 threads per head) where the device allows
     * it, else 2 or 4 (Apple's OpenCL, for one, caps a work-group at 256).
     */
    private int slstmGatesPerThread(int dh) {
        long max = device().getDeviceMaxWorkgroupDimensions()[0];
        for (int gpt = 1; gpt < 4; gpt *= 2) {
            if (4L * dh / gpt <= max) {
                return gpt;
            }
        }
        return 4;
    }

    private uk.ac.manchester.tornado.api.common.TornadoDevice device;

    private uk.ac.manchester.tornado.api.common.TornadoDevice device() {
        if (device == null) {
            String prop = System.getProperty("tirex.tornado.device");
            if (prop != null && !prop.isBlank()) {
                String[] dd = prop.split(":");
                device = TornadoExecutionPlan.getDevice(Integer.parseInt(dd[0]), Integer.parseInt(dd[1]));
            } else {
                device = uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
            }
        }
        return device;
    }

    @Override
    public float[][][] forward(ModelBatch batch) {
        Structure s = structure(batch);
        Shape shape = s.shape(batch.rows());
        Plan plan = plans.get(shape);
        if (plan == null) {
            // evict before building, so the old plan's device buffers are freed first
            while (!plans.isEmpty() && plans.size() >= maxCachedPlans) {
                Shape eldest = plans.keySet().iterator().next();
                plans.remove(eldest).close();
            }
            plan = new Plan(shape);
            plans.put(shape, plan);
        }
        return plan.run(batch, s);
    }

    /** The shapes compiled so far, least recently used first. */
    public List<Shape> cachedShapes() {
        return List.copyOf(plans.keySet());
    }

    /**
     * Known covariates (covariate rows with an observation in the last {@code futureLen} steps) and
     * the rows that share their group with another row. Groups must be contiguous runs of rows, as
     * the batch assembly builds them (one run per series).
     */
    Structure structure(ModelBatch batch) {
        int rows = batch.rows();
        int len = config.paddedLength();
        float[] group = batch.group();
        List<Integer> known = new ArrayList<>();
        for (int r = 0; r < rows; r++) {
            if (batch.target()[r]) {
                continue;
            }
            for (int t = len - config.futureLen(); t < len; t++) {
                if (!Float.isNaN(batch.x()[r][t])) {
                    known.add(r);
                    break;
                }
            }
        }
        List<Integer> kept = new ArrayList<>();
        List<Integer> start = new ArrayList<>(), end = new ArrayList<>(), target = new ArrayList<>();
        java.util.Set<Float> seen = new java.util.HashSet<>();
        for (int r = 0; r < rows; ) {
            int e = r;
            while (e < rows && group[e] == group[r]) {
                e++;
            }
            if (!seen.add(group[r])) {
                throw new IllegalArgumentException("rows of group " + group[r] + " are not contiguous");
            }
            if (e - r > 1) {
                int s0 = kept.size();
                for (int q = r; q < e; q++) {
                    kept.add(q);
                    start.add(s0);
                    end.add(s0 + (e - r));
                    target.add(batch.target()[q] ? 1 : 0);
                }
            }
            r = e;
        }
        return new Structure(ints(known), ints(kept), ints(start), ints(end), ints(target));
    }

    private static int[] ints(List<Integer> v) {
        return v.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public void close() {
        plans.values().forEach(Plan::close);
        plans.clear();
    }

    /** The buffers and execution plan for one shape. */
    private final class Plan implements AutoCloseable {
        final Shape shape;
        final FloatArray x, present, tok, stats, h, n, cell, proj, big, y, out;
        final FloatArray yc, revIn, revOut, xk;
        final IntArray known, kept, groupStart, groupEnd, keptTarget;
        final TornadoExecutionPlan exec;
        final GridScheduler grid;
        final int graphCount;

        Plan(Shape shape) {
            this.shape = shape;
            TiRex2Config c = config;
            int rows = shape.rows(), nKnown = shape.known(), nKept = shape.kept();
            int len = c.paddedLength();
            int seq = c.numPatches();
            int d = c.embeddingDim();
            int p = c.patchSize();
            int nq = c.numQuantiles();
            int tokens = rows * seq;
            int ffn = c.ffnDim();
            int mlstmProj = weights.blocks().stream().filter(PackedWeights.Block::mlstm).findFirst()
                    .map(b -> b.mlstmProj().out()).orElse(0);
            int projWidth = Math.max(Math.max(mlstmProj, ffn), 3 * d);
            int bigWidth = Math.max(Math.max(2 * ffn, 4 * d),
                    Math.max(Math.max(c.inputFfDim(), c.outputFfDim()), TiRex2Config.ATTENTION_FFN_FACTOR * d));

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
            yc = nKnown > 0 ? new FloatArray(tokens * d) : null;
            revIn = nKnown > 0 ? new FloatArray(nKnown * seq * d) : null;
            revOut = nKnown > 0 ? new FloatArray(nKnown * seq * d) : null;
            xk = nKept > 0 ? new FloatArray(nKept * seq * d) : null;
            known = nKnown > 0 ? new IntArray(nKnown) : null;
            kept = nKept > 0 ? new IntArray(nKept) : null;
            groupStart = nKept > 0 ? new IntArray(nKept) : null;
            groupEnd = nKept > 0 ? new IntArray(nKept) : null;
            keptTarget = nKept > 0 ? new IntArray(nKept) : null;
            FloatArray w = weights.data();

            KernelContext ctx = new KernelContext();
            GridScheduler grid = new GridScheduler();
            List<Object> sharedList = new ArrayList<>(List.of(w, x, present, tok, stats, h, n, cell, proj, big, y, out));
            for (Object o : new Object[] {yc, revIn, revOut, xk, known, kept, groupStart, groupEnd, keptTarget}) {
                if (o != null) {
                    sharedList.add(o);
                }
            }
            Object[] shared = sharedList.toArray();
            List<Object> perCall = new ArrayList<>(List.of(x, present));
            for (Object o : new Object[] {known, kept, groupStart, groupEnd, keptTarget}) {
                if (o != null) {
                    perCall.add(o);
                }
            }
            Builder b = new Builder(grid, ctx, w, shared);

            // TornadoVM caps a task graph at 256 tasks and an 8 KiB bytecode buffer, so the forward
            // pass is a chain of graphs (input, one per block, output) sharing device buffers.
            TaskGraph g = b.begin("in");
            g.transferToDevice(DataTransferMode.FIRST_EXECUTION, shared);
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, perCall.toArray());
            // scale always has a worker grid, so every graph registers one: TornadoVM 7.0.x's
            // isGridRegistered() dereferences a null scheduler for a graph with none (the input
            // graph under cuBLAS).
            grid.addWorkerGrid(b.name + ".scale", group(rows * TiRexKernels.NORM_THREADS, TiRexKernels.NORM_THREADS));
            g.task("scale", TiRexKernels::scalePatch, ctx, x, present, tok, stats, rows, len, p, c.useArcsinh() ? 1 : 0,
                    c.binaryAware() ? 1 : 0, TiRex2Config.SCALER_EPS);
            b.residual("in", tok, big, h, tokens, weights.inputEmbedding());
            b.end(false);

            for (int i = 0; i < weights.blocks().size(); i++) {
                PackedWeights.Block blk = weights.blocks().get(i);
                String s = "b" + i + "_";
                g = b.begin("b" + i);
                b.rms(s + "norm", h, n, blk.normLstm(), tokens, d, TiRex2Config.NORM_EPS);
                if (nKnown > 0) {
                    // the reverse pass first, while n still holds RMSNorm(h)
                    b.items(s + "revIn", nKnown * seq * d);
                    g.task(s + "revIn", TiRexKernels::gatherRows, ctx, n, revIn, known, nKnown, seq, d, 1);
                    cell(b, s + "rev_", blk, revIn, revOut, nKnown, false);
                    cell(b, s, blk, n, yc, rows, false);
                    b.items(s + "cat", nKnown * seq * 2 * d);
                    g.task(s + "cat", TiRexKernels::concatForward, ctx, yc, revOut, known, big, nKnown, seq, d);
                    b.gemm(s + "comb", big, blk.combination(), cell, nKnown * seq, 0);
                    b.items(s + "scatter", nKnown * seq * d);
                    g.task(s + "scatter", TiRexKernels::scatterRows, ctx, cell, yc, known, nKnown, seq, d);
                    b.items(s + "resid", tokens * d);
                    g.task(s + "resid", TiRexKernels::addInPlace, ctx, h, yc, tokens * d);
                } else {
                    cell(b, s, blk, n, h, rows, true);
                }
                b.rms(s + "fnorm", h, n, blk.normFfn(), tokens, d, TiRex2Config.NORM_EPS);
                b.gemm(s + "up", n, blk.upGateUp(), big, tokens, 0);
                b.items(s + "swiglu", tokens * blk.ffnDim());
                g.task(s + "swiglu", TiRexKernels::swiGlu, ctx, big, proj, tokens, blk.ffnDim());
                b.gemm(s + "down", proj, blk.down(), h, tokens, TiRexKernels.ACCUMULATE);
                if (nKept > 0) {
                    variate(b, s, blk.variate(), nKept);
                }
                b.end(false);
            }

            g = b.begin("out");
            b.rms("onorm", h, n, weights.stackOutNorm(), tokens, d, c.stackOutNormEps());
            b.residual("out", n, big, y, tokens, weights.outputEmbedding());
            b.items("unpatch", rows * nq * len);
            g.task("unpatch", TiRexKernels::unpatch, ctx, y, stats, out, rows, seq, nq, p, c.useArcsinh() ? 1 : 0,
                    TiRex2Config.SINH_CLAMP);
            g.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            b.end(true);

            graphCount = b.graphs.size();
            this.grid = grid;
            exec = new TornadoExecutionPlan(b.graphs.toArray(new ImmutableTaskGraph[0]));
            String device = System.getProperty("tirex.tornado.device");
            if (device != null && !device.isBlank()) {
                String[] dd = device.split(":");
                exec.withDevice(TornadoExecutionPlan.getDevice(Integer.parseInt(dd[0]), Integer.parseInt(dd[1])));
            }
            if (Boolean.getBoolean("tirex.tornado.cudaGraph")) {
                // CUDA backend only: each task graph is captured once and replayed as a CUDA graph
                exec.withAllGraphs().withCUDAGraph();
            }
        }

        /**
         * One cell pass over {@code count} rows of {@code in} (already RMS-normalised) into
         * {@code dst}: the mLSTM (projection, cell, gated head norm, out projection) or the sLSTM
         * (gates, recurrence, head norm). With {@code add} the output is added to {@code dst},
         * otherwise stored. The mLSTM path overwrites {@code in} with the gated head norm.
         */
        private void cell(Builder b, String s, PackedWeights.Block blk, FloatArray in, FloatArray dst, int count,
                boolean add) {
            TiRex2Config c = config;
            int seq = c.numPatches();
            int d = c.embeddingDim();
            int tokens = count * seq;
            if (blk.mlstm()) {
                PackedWeights.Lin pr = blk.mlstmProj();
                int heads = c.timeMixerHeads();
                int qk = c.mlstmQkDim();
                if (qk / heads != 64 || d / heads != 128) {
                    throw new IllegalStateException("mlstmCell assumes head dims 64/128");
                }
                b.gemm(s + "proj", in, pr, proj, tokens, TiRexKernels.BIAS);
                if (seq > TiRexKernels.MLSTM_MAX_SEQ) {
                    throw new IllegalStateException("mlstmCell supports up to " + TiRexKernels.MLSTM_MAX_SEQ + " patches");
                }
                b.grid.addWorkerGrid(b.name + "." + s + "mlstm", group(count * heads * seq * (d / heads), d / heads));
                b.g.task(s + "mlstm", TiRexKernels::mlstmCell, b.ctx, proj, cell, count, seq, heads, pr.out(), qk, 2 * qk,
                        2 * qk + 2 * d, 2 * qk + 2 * d + heads, TiRex2Config.GATE_SOFT_CAP, TiRex2Config.MLSTM_EPS);
                b.headNorm(s + "hnorm", cell, in, blk.headNorm(), tokens, heads, d / heads, proj, pr.out(), 2 * qk + d, 1);
                b.gemm(s + "out", in, blk.outProj(), dst, tokens, add ? TiRexKernels.ACCUMULATE : 0);
            } else {
                int heads = c.slstmHeads();
                int dh = d / heads;
                b.gemm(s + "gates", in, blk.slstmGates(), big, tokens, 0);
                int gpt = slstmGatesPerThread(dh);
                int threads = 4 * dh / gpt;
                b.grid.addWorkerGrid(b.name + "." + s + "slstm", group(count * heads * threads, threads));
                b.g.task(s + "slstm", TiRexKernels::slstmRecurrence, b.ctx, big, b.w, blk.recurrent(), blk.slstmBias(),
                        cell, seq, heads, gpt);
                b.headNorm(s + "hnorm", cell, dst, blk.groupNorm(), tokens, heads, dh, proj, 0, 0, add ? 2 : 0);
            }
        }

        /** The variate mixer over the {@code nk} kept rows: gather, attention + MLP, scatter back. */
        private void variate(Builder b, String s, PackedWeights.Variate v, int nk) {
            TiRex2Config c = config;
            int seq = c.numPatches();
            int d = c.embeddingDim();
            int heads = c.variateHeads();
            int dh = d / heads;
            int tokens = nk * seq;
            b.items(s + "vgather", tokens * d);
            b.g.task(s + "vgather", TiRexKernels::gatherRows, b.ctx, h, xk, kept, nk, seq, d, 0);
            b.rms(s + "vnorm", xk, n, v.normAttn(), tokens, d, TiRex2Config.NORM_EPS);
            b.gemm(s + "qkv", n, v.qkv(), proj, tokens, 0);
            b.items(s + "qknorm", tokens * 2 * heads);
            b.g.task(s + "qknorm", TiRexKernels::qkNorm, b.ctx, proj, tokens, heads, dh, 3 * d, Math.ulp(1.0f));
            b.items(s + "attn", nk * seq * heads);
            b.g.task(s + "attn", TiRexKernels::variateAttention, b.ctx, proj, cell, groupStart, groupEnd, keptTarget, nk, seq,
                    heads, dh, 3 * d);
            b.gemm(s + "wo", cell, v.wo(), xk, tokens, TiRexKernels.ACCUMULATE);
            b.rms(s + "vfnorm", xk, n, v.normFfn(), tokens, d, TiRex2Config.NORM_EPS);
            b.gemm(s + "vin", n, v.ffnIn(), big, tokens, TiRexKernels.SILU);
            b.gemm(s + "vout", big, v.ffnOut(), xk, tokens, TiRexKernels.ACCUMULATE);
            b.items(s + "vscatter", tokens * d);
            b.g.task(s + "vscatter", TiRexKernels::scatterRows, b.ctx, xk, h, kept, nk, seq, d);
        }

        float[][][] run(ModelBatch batch, Structure s) {
            int rows = shape.rows();
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
            copy(s.known(), known);
            copy(s.kept(), kept);
            copy(s.groupStart(), groupStart);
            copy(s.groupEnd(), groupEnd);
            copy(s.keptTarget(), keptTarget);
            for (int i = 0; i < graphCount; i++) {
                exec.withGraph(i).withGridScheduler(grid).execute();
            }
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

        private void copy(int[] src, IntArray dst) {
            for (int i = 0; dst != null && i < src.length; i++) {
                dst.set(i, src[i]);
            }
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

    /**
     * Builds the chain of task graphs: {@link #begin} opens graph {@code tirex_<name>} consuming the
     * shared buffers from the previous graph, {@link #end} persists them for the next one. Also adds
     * the recurring task shapes (GEMM, norms, residual blocks) with their worker grids.
     */
    private final class Builder {
        final GridScheduler grid;
        final KernelContext ctx;
        final FloatArray w;
        final Object[] shared;
        final List<ImmutableTaskGraph> graphs = new ArrayList<>();
        TaskGraph g;
        String name;
        String previous;

        Builder(GridScheduler grid, KernelContext ctx, FloatArray w, Object[] shared) {
            this.grid = grid;
            this.ctx = ctx;
            this.w = w;
            this.shared = shared;
        }

        final java.util.Set<Object> graphWeights = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());

        TaskGraph begin(String graph) {
            name = G + "_" + graph;
            g = new TaskGraph(name);
            graphWeights.clear();
            if (previous != null) {
                g.consumeFromDevice(previous, shared);
            }
            return g;
        }

        void end(boolean last) {
            if (!last) {
                g.persistOnDevice(shared);
            }
            graphs.add(g.snapshot());
            previous = name;
        }

        /** The worker grid of an element-wise kernel: one thread per item, in groups of {@link #ITEM_GROUP}. */
        void items(String task, int count) {
            grid.addWorkerGrid(this.name + "." + task, group((count + ITEM_GROUP - 1) / ITEM_GROUP * ITEM_GROUP, ITEM_GROUP));
        }

        void gemm(String name, FloatArray a, PackedWeights.Lin l, FloatArray c, int m, int flags) {
            if (gemmMode != GemmMode.TILED) {
                cublasGemm(name, a, l, c, m, flags);
                return;
            }
            int tile = TiRexKernels.TILE;
            WorkerGrid wg = new WorkerGrid2D(ceil(l.out(), tile), ceil(m, tile));
            wg.setLocalWork(tile, tile, 1);
            grid.addWorkerGrid(this.name + "." + name, wg);
            int f = flags | (l.b() >= 0 ? TiRexKernels.BIAS : 0);
            g.task(name, TiRexKernels::gemm, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, 0, m, l.out(), l.in(), f);
        }

        /**
         * {@code C = act(A·Wᵀ + b) (+ C)} with cuBLAS. Row-major {@code C[m, n] = A[m, k]·W[n, k]ᵀ}
         * is the column-major {@code Cᵀ = op_T(W_cm)·A_cm}, i.e. {@code sgemm(T, N, n, m, k, W, k, A,
         * k, C, n)}. The bias is written into C first (then β = 1); SiLU runs afterwards.
         */
        private void cublasGemm(String name, FloatArray a, PackedWeights.Lin l, FloatArray c, int m, int flags) {
            FloatArray wm = weights.matrix(l);
            if (graphWeights.add(wm)) {
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, wm);
            }
            boolean bias = l.b() >= 0;
            boolean acc = (flags & TiRexKernels.ACCUMULATE) != 0;
            if (bias) {
                items(name + "_bias", m * l.out());
                g.task(name + "_bias", TiRexKernels::biasFill, ctx, c, w, l.b(), m, l.out(), acc ? 1 : 0);
            }
            float beta = bias || acc ? 1f : 0f;
            int n = l.out(), k = l.in();
            if (gemmMode == GemmMode.CUBLAS_TF32) {
                g.libraryTask(name, uk.ac.manchester.tornado.cublas.CuBlas::cublasSgemmTF32, 1, 0, n, m, k, 1f, wm, k, a, k,
                        beta, c, n);
            } else {
                g.libraryTask(name, uk.ac.manchester.tornado.cublas.CuBlas::cublasSgemm, 1, 0, n, m, k, 1f, wm, k, a, k,
                        beta, c, n);
            }
            if ((flags & TiRexKernels.SILU) != 0) {
                items(name + "_silu", m * n);
                g.task(name + "_silu", TiRexKernels::siluInPlace, ctx, c, m * n);
            }
        }

        void rms(String name, FloatArray x, FloatArray y, int wOff, int rows, int d, float eps) {
            grid.addWorkerGrid(this.name + "." + name, group(rows * TiRexKernels.NORM_THREADS, TiRexKernels.NORM_THREADS));
            g.task(name, TiRexKernels::rmsNorm, ctx, x, y, w, wOff, d, eps);
        }

        void headNorm(String name, FloatArray x, FloatArray y, int wOff, int tokens, int heads, int dh, FloatArray gate,
                int gStride, int gOff, int mode) {
            grid.addWorkerGrid(this.name + "." + name, group(tokens * heads * dh, dh));
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
