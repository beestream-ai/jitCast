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
package ai.beestream.jitcast.chronos2.tornado;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitcast.chronos2.Chronos2Config;
import ai.beestream.jitcast.chronos2.ModelForward;
import ai.beestream.jitcast.chronos2.ModelInput;
import ai.beestream.jitcast.chronos2.ModelLayout;
import ai.beestream.jitcast.chronos2.ModelPrep;
import ai.beestream.jitcast.chronos2.cpu.CpuChronos2;
import ai.beestream.jitcast.tornado.GemmMode;
import ai.beestream.jitcast.tornado.Kernels;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cublas.CuBlas;

/**
 * {@code Chronos2Model.forward} on a TornadoVM device: the instance norm and the patch tokens, and
 * the inverse norm run on the host ({@link ModelPrep}); the input patch embedding, the encoder and
 * the output patch embedding run as one task graph per shape (rows, context patches, output
 * patches), compiled once and cached. The rows are reordered so that each group is a contiguous run
 * for the group attention, and put back in the input's order after.
 *
 * <p>The device is {@code -Djitcast.tornado.device=<driver>:<device>}, or the default one. GEMMs run
 * as {@code -Dchronos2.tornado.gemm} says ({@link GemmMode}; {@code auto} by default: cuBLAS in
 * FP32 on CUDA, the register-blocked FP32 kernel elsewhere). The blocked kernel takes GEMMs from
 * {@code -Dchronos2.tornado.blockedMinRows} rows (128 by default) and the tiled kernel the smaller
 * ones. With cuBLAS the task graph is captured and replayed as a CUDA graph
 * ({@code -Dchronos2.tornado.cudaGraph}, on by default with cuBLAS).
 */
public final class Chronos2Forward implements ModelForward, AutoCloseable {

    private static final String G = "c2";

    /** Work-group size of the element-wise kernels. */
    private static final int ITEM_GROUP = 128;

    /** The fewest rows a GEMM needs to run on {@link Kernels#gemmBlocked}. */
    private final int blockedMinRows = Integer.getInteger("chronos2.tornado.blockedMinRows", 128);

    /** The fewest rows a GEMM needs to run on {@link Kernels#gemmBlockedWide}, whose 64 × 128 tiles leave the GPU idle below that. */
    private final int wideMinRows = Integer.getInteger("chronos2.tornado.wideMinRows", 1024);

    /** One compiled shape. */
    public record Shape(int rows, int contextPatches, int outputPatches) {}

    private final Chronos2Config config;
    private final Chronos2Packed weights;
    private final float[] invFreq;
    private final int maxCachedPlans;
    private final Map<Shape, Plan> plans = new LinkedHashMap<>(16, 0.75f, true);
    private TornadoDevice device;
    private final GemmMode gemmMode;
    private final boolean cudaGraph;

    public Chronos2Forward(Chronos2Config config, Chronos2Packed weights) {
        this(config, weights, Integer.getInteger("chronos2.tornado.plans", 8));
    }

    public Chronos2Forward(Chronos2Config config, Chronos2Packed weights, int maxCachedPlans) {
        if (config.dKv() > Chronos2Kernels.BLOCK_HEAD_DIM || config.dKv() % 16 != 0) {
            throw new IllegalArgumentException("unsupported attention head width " + config.dKv());
        }
        this.config = config;
        this.weights = weights;
        this.invFreq = CpuChronos2.inverseFrequencies(config);
        this.maxCachedPlans = Math.max(1, maxCachedPlans);
        this.gemmMode = GemmMode.parse(System.getProperty("chronos2.tornado.gemm", "auto")).resolve(device());
        this.cudaGraph = Boolean.parseBoolean(System.getProperty("chronos2.tornado.cudaGraph", String.valueOf(gemmMode.cublas())));
    }

    /** The GEMM mode in use, {@code AUTO} resolved for the device. */
    public GemmMode gemmMode() {
        return gemmMode;
    }

    private TornadoDevice device() {
        if (device == null) {
            String prop = System.getProperty("jitcast.tornado.device");
            if (prop != null && !prop.isBlank()) {
                String[] dd = prop.split(":");
                device = TornadoExecutionPlan.getDevice(Integer.parseInt(dd[0]), Integer.parseInt(dd[1]));
            } else {
                device = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
            }
        }
        return device;
    }

    /** The shapes compiled so far, least recently used first. */
    public List<Shape> cachedShapes() {
        return List.copyOf(plans.keySet());
    }

    @Override
    public float[][][] forward(ModelInput input) {
        int[] order = groupOrder(input.group());
        ModelInput in = input;
        if (order != null) {
            float[][] c = new float[order.length][], f = input.futureCovariates() == null ? null : new float[order.length][];
            int[] g = new int[order.length];
            for (int i = 0; i < order.length; i++) {
                c[i] = input.context()[order[i]];
                g[i] = input.group()[order[i]];
                if (f != null) {
                    f[i] = input.futureCovariates()[order[i]];
                }
            }
            in = new ModelInput(c, g, f, input.outputPatches());
        }
        ModelLayout lay = ModelLayout.of(config, in);
        ModelPrep prep = ModelPrep.of(config, lay);
        Shape shape = new Shape(lay.rows(), lay.contextPatches(), lay.outputPatches());
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
        float[][][] out = prep.finish(config, lay, plan.run(lay, prep));
        if (order == null) {
            return out;
        }
        float[][][] back = new float[out.length][][];
        for (int i = 0; i < order.length; i++) {
            back[order[i]] = out[i];
        }
        return back;
    }

    /** The rows grouped into contiguous runs (groups in order of first appearance), or null if they already are. */
    static int[] groupOrder(int[] group) {
        Map<Integer, List<Integer>> byGroup = new LinkedHashMap<>();
        for (int i = 0; i < group.length; i++) {
            byGroup.computeIfAbsent(group[i], g -> new ArrayList<>()).add(i);
        }
        int[] order = new int[group.length];
        int k = 0;
        boolean identity = true;
        for (List<Integer> rows : byGroup.values()) {
            for (int r : rows) {
                identity &= r == k;
                order[k++] = r;
            }
        }
        return identity ? null : order;
    }

    @Override
    public void close() {
        plans.values().forEach(Plan::close);
        plans.clear();
    }

    private static WorkerGrid group(int global, int local) {
        WorkerGrid wg = new WorkerGrid1D(global);
        wg.setLocalWork(local, 1, 1);
        return wg;
    }

    /** The buffers and execution plan for one shape. */
    private final class Plan implements AutoCloseable {
        final Shape shape;
        final FloatArray tok, emb, x, hn, big, att, y, dec, raw, table;
        final IntArray live, groupStart, groupEnd;
        final HalfFloatArray noTwin = new HalfFloatArray(1);
        final TornadoExecutionPlan exec;
        final GridScheduler grid = new GridScheduler();
        final KernelContext ctx = new KernelContext();
        final FloatArray w = weights.data();
        TaskGraph g;
        /** FP16 copies of GEMM inputs, and the ones whose copy is current (no task has written the source since). */
        final Map<FloatArray, HalfFloatArray> halves = new IdentityHashMap<>();
        final Set<FloatArray> fresh = Collections.newSetFromMap(new IdentityHashMap<>());
        /** The per-matrix weight copies already added to the graph. */
        final Set<Object> graphWeights = Collections.newSetFromMap(new IdentityHashMap<>());

        Plan(Shape shape) {
            this.shape = shape;
            Chronos2Config c = config;
            int rows = shape.rows(), nc = shape.contextPatches(), h = shape.outputPatches();
            int reg = c.useRegToken() ? 1 : 0, n = nc + reg + h, ne = nc + h;
            int d = c.dModel(), inner = c.inner(), ff = c.dFf(), ps = c.patchSize(), q = c.numQuantiles();
            int heads = c.numHeads(), dh = c.dKv();
            int tokens = rows * n, embTokens = rows * ne, futTokens = rows * h;
            tok = new FloatArray(embTokens * 3 * ps);
            emb = new FloatArray(embTokens * d);
            x = new FloatArray(tokens * d);
            hn = new FloatArray(tokens * d);
            big = new FloatArray(tokens * Math.max(3 * inner, ff));
            att = new FloatArray(tokens * inner);
            y = new FloatArray(tokens * d);
            dec = new FloatArray(futTokens * d);
            raw = new FloatArray(futTokens * q * ps);
            table = new FloatArray(n * dh);
            live = new IntArray(tokens);
            groupStart = new IntArray(rows);
            groupEnd = new IntArray(rows);

            g = new TaskGraph(G);
            g.transferToDevice(DataTransferMode.FIRST_EXECUTION, w, emb, x, hn, big, att, y, dec, raw, noTwin);
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, tok, table, live, groupStart, groupEnd);

            residual("emb", tok, big, emb, embTokens, weights.inputEmbedding());
            items("assemble", tokens * d);
            g.task("assemble", Chronos2Kernels::assembleTokens, ctx, emb, w, Math.max(weights.reg(), 0), x, rows, nc, reg, n, d);
            written(x);

            int qTiles = (n + Chronos2Kernels.TIME_QUERIES - 1) / Chronos2Kernels.TIME_QUERIES;
            int rTiles = (rows + Chronos2Kernels.ATTN_QUERIES - 1) / Chronos2Kernels.ATTN_QUERIES;
            for (int li = 0; li < weights.layers().size(); li++) {
                Chronos2Packed.Layer layer = weights.layers().get(li);
                String s = "l" + li + "t_";
                rms(s + "norm", x, hn, layer.time().norm(), tokens);
                gemm(s + "qkv", hn, layer.time().qkv(), big, tokens, 0);
                items(s + "rope", tokens * heads * (dh / 2));
                g.task(s + "rope", Chronos2Kernels::rope, ctx, big, table, tokens, n, heads, dh, inner);
                written(big);
                grid.addWorkerGrid(G + "." + s + "attn", group(rows * heads * qTiles * Chronos2Kernels.ATTN_THREADS,
                        Chronos2Kernels.ATTN_THREADS));
                g.task(s + "attn", Chronos2Kernels::timeAttention, ctx, big, att, live, rows, n, heads, dh, inner);
                written(att);
                gemm(s + "o", att, layer.time().o(), x, tokens, Kernels.ACCUMULATE);

                s = "l" + li + "g_";
                rms(s + "norm", x, hn, layer.group().norm(), tokens);
                gemm(s + "qkv", hn, layer.group().qkv(), big, tokens, 0);
                grid.addWorkerGrid(G + "." + s + "attn", group(n * heads * rTiles * Chronos2Kernels.ATTN_THREADS,
                        Chronos2Kernels.ATTN_THREADS));
                g.task(s + "attn", Chronos2Kernels::groupAttention, ctx, big, att, live, groupStart, groupEnd, rows, n, heads, dh,
                        inner);
                written(att);
                gemm(s + "o", att, layer.group().o(), x, tokens, Kernels.ACCUMULATE);

                s = "l" + li + "f_";
                rms(s + "norm", x, hn, layer.mlpNorm(), tokens);
                gemm(s + "wi", hn, layer.wi(), big, tokens, Kernels.RELU);
                gemm(s + "wo", big, layer.wo(), x, tokens, Kernels.ACCUMULATE);
            }
            rms("fnorm", x, y, weights.finalNorm(), tokens);
            items("gather", futTokens * d);
            g.task("gather", Chronos2Kernels::gatherFuture, ctx, y, dec, rows, nc + reg, n, h, d);
            written(dec);
            residual("out", dec, big, raw, futTokens, weights.outputEmbedding());
            g.transferToHost(DataTransferMode.EVERY_EXECUTION, raw);

            exec = new TornadoExecutionPlan(g.snapshot());
            exec.withGridScheduler(grid);
            exec.withDevice(device());
            if (cudaGraph) {
                // CUDA backend only: the task graph is captured once and replayed as a CUDA graph
                exec.withCUDAGraph();
            }
            if (Boolean.getBoolean("chronos2.tornado.profile")) {
                exec.withProfiler(uk.ac.manchester.tornado.api.enums.ProfilerMode.CONSOLE);
            }
        }

        /** {@code output(ReLU(hidden(a))) + residual(a)}, with the hidden activation in {@code scratch}. */
        void residual(String name, FloatArray a, FloatArray scratch, FloatArray out, int m, Chronos2Packed.Residual r) {
            gemm(name + "_h", a, r.hidden(), scratch, m, Kernels.RELU);
            gemm(name + "_o", scratch, r.output(), out, m, 0);
            gemm(name + "_r", a, r.residual(), out, m, Kernels.ACCUMULATE);
        }

        /** {@code C = act(A·Wᵀ + b) (+ C)}, with the flags of {@link Kernels#gemm}, in the plan's GEMM mode. */
        void gemm(String name, FloatArray a, Chronos2Packed.Lin l, FloatArray c, int m, int flags) {
            if (gemmMode.cublas()) {
                cublas(name, a, l, c, m, flags);
                written(c);
                return;
            }
            int f = flags | (l.b() >= 0 ? Kernels.BIAS : 0);
            WorkerGrid wg;
            if (gemmMode == GemmMode.HALF) {
                HalfFloatArray ah = half(name, a, m * l.in());
                HalfFloatArray wh = halfWeight(l);
                int t = Kernels.HGEMM_TILE, threads = Kernels.HGEMM_THREADS;
                wg = new WorkerGrid2D((l.out() + t - 1) / t * threads, (m + t - 1) / t * threads);
                wg.setLocalWork(threads, threads, 1);
                g.task(name, Kernels::gemmHalf, ctx, ah, wh, c, w, Math.max(l.b(), 0), m, l.out(), l.in(), f, noTwin, 0);
            } else if (gemmMode == GemmMode.BLOCKED && m >= wideMinRows) {
                int tr = Kernels.WIDE_ROWS, tc = Kernels.WIDE_COLS, threads = Kernels.HGEMM_THREADS;
                wg = new WorkerGrid2D((l.out() + tc - 1) / tc * threads, (m + tr - 1) / tr * threads);
                wg.setLocalWork(threads, threads, 1);
                g.task(name, Kernels::gemmBlockedWide, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, 0, m, l.out(), l.in(), f);
            } else if (gemmMode == GemmMode.BLOCKED && m >= blockedMinRows) {
                int t = Kernels.HGEMM_TILE, threads = Kernels.HGEMM_THREADS;
                wg = new WorkerGrid2D((l.out() + t - 1) / t * threads, (m + t - 1) / t * threads);
                wg.setLocalWork(threads, threads, 1);
                g.task(name, Kernels::gemmBlocked, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, 0, m, l.out(), l.in(), f);
            } else {
                int tile = Kernels.TILE;
                wg = new WorkerGrid2D((l.out() + tile - 1) / tile * tile, (m + tile - 1) / tile * tile);
                wg.setLocalWork(tile, tile, 1);
                g.task(name, Kernels::gemm, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, 0, m, l.out(), l.in(), f);
            }
            grid.addWorkerGrid(G + "." + name, wg);
            written(c);
        }

        /**
         * The GEMM in cuBLAS. Row-major {@code C[m, n] = A[m, k]·W[n, k]ᵀ} is the column-major
         * {@code Cᵀ = op_T(W)·A}, i.e. {@code gemm(T, N, n, m, k, W, k, A, k, C, n)}. A bias is written
         * into C first (then β = 1), except before a ReLU, where one epilogue adds the bias and
         * applies the ReLU after the GEMM.
         */
        private void cublas(String name, FloatArray a, Chronos2Packed.Lin l, FloatArray c, int m, int flags) {
            int n = l.out(), k = l.in();
            boolean bias = l.b() >= 0;
            boolean acc = (flags & Kernels.ACCUMULATE) != 0;
            boolean relu = (flags & Kernels.RELU) != 0;
            if (relu && acc) {
                throw new IllegalStateException("GEMM " + name + ": ReLU after accumulation is not supported");
            }
            if (bias && !relu) {
                items(name + "_bias", m * n);
                g.task(name + "_bias", Kernels::biasFill, ctx, c, w, l.b(), m, n, acc ? 1 : 0);
            }
            float beta = (bias && !relu) || acc ? 1f : 0f;
            if (gemmMode == GemmMode.CUBLAS_FP16) {
                HalfFloatArray ah = half(name, a, m * k);
                HalfFloatArray wh = halfWeight(l);
                g.libraryTask(name, CuBlas::cublasGemmExFP16FP32, 1, 0, n, m, k, 1f, wh, k, ah, k, beta, c, n);
            } else {
                FloatArray wm = weights.matrix(l);
                if (graphWeights.add(wm)) {
                    g.transferToDevice(DataTransferMode.FIRST_EXECUTION, wm);
                }
                if (gemmMode == GemmMode.CUBLAS_TF32) {
                    g.libraryTask(name, CuBlas::cublasSgemmTF32, 1, 0, n, m, k, 1f, wm, k, a, k, beta, c, n);
                } else {
                    g.libraryTask(name, CuBlas::cublasSgemm, 1, 0, n, m, k, 1f, wm, k, a, k, beta, c, n);
                }
            }
            if (relu) {
                items(name + "_relu", m * n);
                g.task(name + "_relu", Kernels::biasRelu, ctx, c, w, bias ? l.b() : -1, m, n, noTwin, 0);
            }
        }

        /** The FP16 copy of a GEMM input, converted once after each task that writes the source. */
        private HalfFloatArray half(String name, FloatArray a, int count) {
            HalfFloatArray h = halves.get(a);
            if (h == null) {
                h = new HalfFloatArray(a.getSize());
                halves.put(a, h);
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, h);
            }
            if (fresh.add(a)) {
                items(name + "_h", count);
                g.task(name + "_h", Kernels::toHalf, ctx, a, h, count);
            }
            return h;
        }

        private HalfFloatArray halfWeight(Chronos2Packed.Lin l) {
            HalfFloatArray wh = weights.halfMatrix(l);
            if (graphWeights.add(wh)) {
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, wh);
            }
            return wh;
        }

        /** A task has written {@code buf}: its FP16 copy is stale. */
        private void written(FloatArray buf) {
            fresh.remove(buf);
        }

        void rms(String name, FloatArray in, FloatArray out, int wOff, int rows) {
            grid.addWorkerGrid(G + "." + name, group(rows * Kernels.NORM_THREADS, Kernels.NORM_THREADS));
            g.task(name, Kernels::rmsNorm, ctx, in, out, w, wOff, config.dModel(), config.layerNormEps(), noTwin, 0);
            written(out);
        }

        void items(String task, int count) {
            grid.addWorkerGrid(G + "." + task, group((count + ITEM_GROUP - 1) / ITEM_GROUP * ITEM_GROUP, ITEM_GROUP));
        }

        /** Uploads one pass's per-call inputs, runs the graph, returns the output patch embedding's raw outputs. */
        float[] run(ModelLayout lay, ModelPrep prep) {
            int rows = shape.rows(), n = lay.tokens(), dh = config.dKv();
            float[] t = prep.tokens();
            MemorySegment.copy(t, 0, tok.getSegment(), ValueLayout.JAVA_FLOAT, 0, t.length);
            int half = dh / 2;
            for (int p = 0; p < n; p++) {
                for (int e = 0; e < half; e++) {
                    float a = (float) p * invFreq[e];
                    table.set(p * dh + e, (float) Math.cos(a));
                    table.set(p * dh + half + e, (float) Math.sin(a));
                }
            }
            for (int i = 0; i < rows; i++) {
                for (int tk = 0; tk < n; tk++) {
                    live.set(i * n + tk, lay.live(i, tk) ? 1 : 0);
                }
                if (i == 0 || lay.group(i) != lay.group(i - 1)) {
                    int e = i;
                    while (e < rows && lay.group(e) == lay.group(i)) {
                        e++;
                    }
                    for (int k = i; k < e; k++) {
                        groupStart.set(k, i);
                        groupEnd.set(k, e);
                    }
                }
            }
            exec.execute();
            float[] out = new float[raw.getSize()];
            MemorySegment.copy(raw.getSegment(), ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
            return out;
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
}
