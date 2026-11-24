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
package ai.beestream.jitcast.timesfm.tornado;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitcast.timesfm.PatchForward;
import ai.beestream.jitcast.timesfm.TimesFmConfig;
import ai.beestream.jitcast.tornado.Devices;
import ai.beestream.jitcast.tornado.GemmMode;
import ai.beestream.jitcast.tornado.Kernels;
import ai.beestream.jitcast.tornado.PlanCache;
import ai.beestream.jitcast.util.Options;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cublas.CuBlas;

/**
 * TimesFM 2.5's forward pass on a TornadoVM device: the tokenizer, the transformer layers and the heads
 * of each row's last patch, as one task graph per shape (rows, patches, whether the continuous quantile
 * head runs), compiled once and cached. The host builds the tokens (each patch's values and mask) and the
 * count of leading padding patches per row.
 *
 * <p>The device is {@code -Djitcast.tornado.device=<backend>:<device>}, or the default one. GEMMs run as
 * {@code -Dtimesfm.tornado.gemm} says ({@link GemmMode}; {@code auto} by default: cuBLAS in FP32 on CUDA,
 * Apple's simdgroup matrices on Metal, the register-blocked FP32 kernel elsewhere). With cuBLAS the task
 * graph is captured and replayed as a CUDA graph ({@code -Dtimesfm.tornado.cudaGraph}, on by default with
 * cuBLAS).
 */
public final class TimesFmForward implements PatchForward {

    static {
        Devices.defaults();
    }

    private static final String G = "tfm";

    /** Work-group size of the element-wise kernels. */
    private static final int ITEM_GROUP = 128;

    /** The fewest rows a GEMM needs to run on {@link Kernels#gemmBlocked}. */
    private final int blockedMinRows = Options.integer("timesfm.tornado.blockedMinRows", 128, 1);

    /** The fewest rows a GEMM needs to run on {@link Kernels#gemmBlockedWide}. */
    private final int blockedWideMinRows = Options.integer("timesfm.tornado.wideMinRows", 1024, 1);

    /** The fewest GEMM rows that use {@link Kernels#gemmSimdgroupWide} rather than {@link Kernels#gemmSimdgroup}. */
    private final int simdgroupWideMinRows = Options.integer("timesfm.tornado.simdgroupWideMinRows", 64, 1);

    /**
     * Split K: a 64 × 64 simdgroup GEMM with fewer tiles than {@code splitKTarget} work-groups also splits
     * K, into at most {@code splitKMax} slices of at least {@link #SPLIT_K_MIN} each. 1 turns it off.
     */
    private final int splitKTarget = Options.integer("timesfm.tornado.splitKTarget", 320, 1);
    private final int splitKMax = Options.integer("timesfm.tornado.splitKMax", 16, 1);
    static final int SPLIT_K_MIN = 128;

    int splitK(int tiles, int k) {
        if (tiles >= splitKTarget) {
            return 1;
        }
        return Math.max(1, Math.min(Math.min(splitKMax, (splitKTarget + tiles - 1) / tiles), k / SPLIT_K_MIN));
    }

    /**
     * One compiled shape.
     *
     * @param rows the sequences
     * @param patches the patches per sequence (the prefill's, when decoding)
     * @param spread whether the continuous quantile head runs
     * @param steps the decoding steps after the prefill; with any, the plan keeps every layer's keys and
     *     values on the device and has a second task graph that runs one step
     */
    public record Shape(int rows, int patches, boolean spread, int steps) {}

    private final TimesFmConfig config;
    private final TimesFmPacked weights;
    private final PlanCache<Shape, Plan> plans;
    private final TornadoDevice device = Devices.device();
    private final GemmMode gemmMode;
    private final boolean cudaGraph;

    /**
     * A forward pass with the default plan count ({@code -Dtimesfm.tornado.plans}, 8).
     *
     * @param config the checkpoint's configuration
     * @param weights the packed weights
     */
    public TimesFmForward(TimesFmConfig config, TimesFmPacked weights) {
        this(config, weights, Options.integer("timesfm.tornado.plans", 8, 1));
    }

    /**
     * A forward pass keeping at most {@code maxCachedPlans} compiled shapes.
     *
     * @param config the checkpoint's configuration
     * @param weights the packed weights
     * @param maxCachedPlans the most compiled shapes to keep
     */
    public TimesFmForward(TimesFmConfig config, TimesFmPacked weights, int maxCachedPlans) {
        if (config.contextLength() / config.patchLength() > TimesFmKernels.ATTN_MAX_KEYS) {
            throw new IllegalArgumentException("sequences longer than " + TimesFmKernels.ATTN_MAX_KEYS + " patches are not supported");
        }
        if (config.headDim() > TimesFmKernels.ATTN_HEAD_DIM || config.headDim() > TimesFmKernels.ROPE_THREADS) {
            throw new IllegalArgumentException("attention heads wider than " + TimesFmKernels.ATTN_HEAD_DIM + " are not supported");
        }
        this.config = config;
        this.weights = weights;
        this.plans = new PlanCache<>(Math.max(1, maxCachedPlans), Options.bytes("timesfm.tornado.planBytes", Devices.memory(), 1));
        this.gemmMode = Options.enumeration("timesfm.tornado.gemm", GemmMode.AUTO, GemmMode.class).resolve(device);
        this.cudaGraph = Options.bool("timesfm.tornado.cudaGraph", gemmMode.cublas());
    }

    /** {@return the GEMM mode in use, {@code AUTO} resolved for the device} */
    public GemmMode gemmMode() {
        return gemmMode;
    }

    /** {@return the shapes compiled so far, least recently used first} */
    public List<Shape> cachedShapes() {
        return plans.keys();
    }

    @Override
    public Heads forward(Batch batch) {
        check(batch.patches());
        Shape shape = new Shape(batch.rows(), batch.patches(), batch.spread(), 0);
        return plans.run(shape, Plan::new, plan -> plan.prefill(batch));
    }

    /**
     * Decodes with a key-value cache: the prefill writes every layer's keys and values to the device, and
     * each step runs only its new patches, attending to the cached ones. One plan per (rows, prefill
     * patches, steps) serves every step.
     */
    @Override
    public List<Heads> decode(Batch prefill, int steps, Feeder feeder) {
        if (steps == 0) {
            return List.of(forward(prefill));
        }
        check(prefill.patches() + steps * config.patchesPerOutput());
        Shape shape = new Shape(prefill.rows(), prefill.patches(), prefill.spread(), steps);
        return plans.run(shape, Plan::new, plan -> {
            List<Heads> out = new ArrayList<>();
            Heads h = plan.prefill(prefill);
            out.add(h);
            for (int s = 0; s < steps; s++) {
                h = plan.step(feeder.next(h, s), prefill.patches() + s * config.patchesPerOutput());
                out.add(h);
            }
            return out;
        });
    }

    private static void check(int patches) {
        if (patches > TimesFmKernels.ATTN_MAX_KEYS) {
            throw new IllegalArgumentException("at most " + TimesFmKernels.ATTN_MAX_KEYS + " patches, got " + patches);
        }
    }

    @Override
    public void close() {
        plans.close();
    }

    private static WorkerGrid group(int global, int local) {
        WorkerGrid wg = new WorkerGrid1D(global);
        wg.setLocalWork(local, 1, 1);
        return wg;
    }

    /** The task graphs, caches and execution plan for one shape. */
    private final class Plan implements PlanCache.Plan {
        final Shape shape;
        final FloatArray w = weights.data();
        final FloatArray kc, vc;
        final int cap;
        final GridScheduler grid = new GridScheduler();
        final KernelContext ctx = new KernelContext();
        /** The weight buffers the prefill graph uploads, which the step graph takes from it. */
        final Set<Object> prefillWeights = Collections.newSetFromMap(new IdentityHashMap<>());
        final Graph pre;
        final Graph step;
        final TornadoExecutionPlan exec;

        Plan(Shape shape) {
            this.shape = shape;
            int m = config.patchesPerOutput();
            cap = shape.patches() + shape.steps() * m;
            long cacheSize = shape.steps() > 0 ? (long) config.numLayers() * shape.rows() * cap * config.hiddenSize() : 1;
            if (cacheSize > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("key-value cache of " + cacheSize + " floats is too large; use smaller batches");
            }
            kc = new FloatArray((int) cacheSize);
            vc = new FloatArray((int) cacheSize);
            pre = new Graph(G, shape.patches(), true, shape.spread(), shape.steps() > 0);
            if (shape.steps() > 0) {
                pre.g.persistOnDevice(prefillWeights.toArray());
                pre.g.persistOnDevice(kc, vc);
                step = new Graph(G + "d", m, false, false, true);
                exec = new TornadoExecutionPlan(pre.g.snapshot(), step.g.snapshot());
            } else {
                step = null;
                exec = new TornadoExecutionPlan(pre.g.snapshot());
            }
            exec.withGridScheduler(grid);
            exec.withDevice(device);
            if (cudaGraph) {
                exec.withCUDAGraph();
            }
            if (Options.bool("timesfm.tornado.profile", false)) {
                exec.withProfiler(uk.ac.manchester.tornado.api.enums.ProfilerMode.CONSOLE);
            }
        }

        Heads prefill(Batch b) {
            pre.upload(b, 0);
            if (step != null) {
                exec.withGraph(0).execute();
            } else {
                exec.execute();
            }
            return pre.heads();
        }

        Heads step(Batch b, int start) {
            step.upload(b, start);
            exec.withGraph(1).execute();
            return step.heads();
        }

        /**
         * One task graph: the prefill over {@code n} patches per row (writing the cache when decoding), or a
         * step over {@code n} new patches attending to the cache.
         */
        private final class Graph {
            final String name;
            final int n;
            final boolean first;
            final boolean spread;
            final TaskGraph g;
            final FloatArray tok, x, hn, big, att, y, dec, hb, pt, sp;
            final IntArray masked, off;
            final HalfFloatArray noTwin = new HalfFloatArray(1);
            FloatArray part;
            final Map<FloatArray, HalfFloatArray> halves = new IdentityHashMap<>();
            final Set<FloatArray> fresh = Collections.newSetFromMap(new IdentityHashMap<>());
            final Set<Object> declared = Collections.newSetFromMap(new IdentityHashMap<>());

            Graph(String name, int n, boolean first, boolean spread, boolean cache) {
                this.name = name;
                this.n = n;
                this.first = first;
                this.spread = spread;
                TimesFmConfig c = config;
                int rows = shape.rows();
                int d = c.hiddenSize(), ff = c.intermediateSize(), p = c.patchLength(), heads = c.numHeads(), hd = c.headDim();
                int tokens = rows * n;
                tok = new FloatArray(tokens * 2 * p);
                x = new FloatArray(tokens * d);
                hn = new FloatArray(tokens * d);
                big = new FloatArray(tokens * Math.max(3 * d, ff));
                att = new FloatArray(tokens * d);
                y = new FloatArray(tokens * d);
                dec = new FloatArray(rows * d);
                hb = new FloatArray(rows * d);
                pt = new FloatArray(rows * c.horizonLength() * c.outputs());
                sp = new FloatArray(spread ? rows * c.quantileHorizonLength() * c.outputs() : 1);
                masked = new IntArray(rows);
                off = new IntArray(1);

                g = new TaskGraph(name);
                weight(w);
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, x, hn, big, att, y, dec, hb, pt, sp, noTwin);
                g.transferToDevice(DataTransferMode.EVERY_EXECUTION, tok, masked, off);
                if (!first) {
                    g.consumeFromDevice(G, kc, vc);
                } else if (cache) {
                    g.transferToDevice(DataTransferMode.FIRST_EXECUTION, kc, vc);
                }

                residual("tok", tok, big, x, tokens, weights.tokenizer());
                for (int li = 0; li < weights.layers().size(); li++) {
                    TimesFmPacked.Layer l = weights.layers().get(li);
                    String s = "l" + li + "_";
                    rms(s + "pa", x, hn, l.preAttn(), tokens);
                    gemm(s + "qkv", hn, l.qkv(), big, tokens, 0);
                    grid.addWorkerGrid(name + "." + s + "rope", group(tokens * heads * TimesFmKernels.ROPE_THREADS, TimesFmKernels.ROPE_THREADS));
                    g.task(s + "rope", TimesFmKernels::ropeNorm, ctx, big, w, l.qNorm(), l.kNorm(), l.qScale(), weights.invTimescale(),
                            masked, off, rows, n, heads, hd, d, c.rmsNormEps());
                    written(big);
                    if (cache) {
                        items(s + "kv", tokens * d);
                        g.task(s + "kv", TimesFmKernels::cacheWrite, ctx, big, kc, vc, off, li, rows, n, cap, d);
                    }
                    if (first) {
                        int tiles = (n + TimesFmKernels.ATTN_QUERIES - 1) / TimesFmKernels.ATTN_QUERIES;
                        grid.addWorkerGrid(name + "." + s + "attn", group(rows * heads * tiles * TimesFmKernels.ATTN_THREADS,
                                TimesFmKernels.ATTN_THREADS));
                        g.task(s + "attn", TimesFmKernels::causalAttention, ctx, big, att, masked, rows, n, heads, hd, d);
                    } else {
                        grid.addWorkerGrid(name + "." + s + "attn", group(rows * heads * TimesFmKernels.ATTN_THREADS, TimesFmKernels.ATTN_THREADS));
                        g.task(s + "attn", TimesFmKernels::cachedAttention, ctx, big, kc, vc, att, masked, off, li, rows, n, cap, heads, hd, d);
                    }
                    written(att);
                    gemm(s + "o", att, l.out(), y, tokens, 0);
                    rms(s + "po", y, hn, l.postAttn(), tokens);
                    add(s + "ra", x, hn, tokens * d);
                    rms(s + "pf", x, hn, l.preFf(), tokens);
                    gemm(s + "ff0", hn, l.ff0(), big, tokens, Kernels.SILU);
                    gemm(s + "ff1", big, l.ff1(), y, tokens, 0);
                    rms(s + "pof", y, hn, l.postFf(), tokens);
                    add(s + "rf", x, hn, tokens * d);
                }
                items("gather", rows * d);
                g.task("gather", TimesFmKernels::gatherLast, ctx, x, dec, rows, n, d);
                written(dec);
                residual("pt", dec, hb, pt, rows, weights.pointHead());
                if (spread) {
                    residual("sp", dec, hb, sp, rows, weights.quantileHead());
                    g.transferToHost(DataTransferMode.EVERY_EXECUTION, pt, sp);
                } else {
                    g.transferToHost(DataTransferMode.EVERY_EXECUTION, pt);
                }
            }

            /** A weight buffer: uploaded once by the prefill graph, taken from it by the step graph. */
            private void weight(Object buf) {
                if (!declared.add(buf)) {
                    return;
                }
                if (first) {
                    g.transferToDevice(DataTransferMode.FIRST_EXECUTION, buf);
                    prefillWeights.add(buf);
                } else if (prefillWeights.contains(buf)) {
                    g.consumeFromDevice(G, buf);
                } else {
                    g.transferToDevice(DataTransferMode.FIRST_EXECUTION, buf);
                }
            }

            void residual(String task, FloatArray a, FloatArray scratch, FloatArray out, int m, TimesFmPacked.Residual r) {
                gemm(task + "_h", a, r.hidden(), scratch, m, Kernels.SILU);
                gemm(task + "_o", scratch, r.output(), out, m, 0);
                gemm(task + "_r", a, r.residual(), out, m, Kernels.ACCUMULATE);
            }

            void add(String task, FloatArray acc, FloatArray v, int count) {
                items(task, count);
                g.task(task, Kernels::addInPlace, ctx, acc, v, count);
                written(acc);
            }

            /** {@code C = act(A·Wᵀ + b) (+ C)}, with the flags of {@link Kernels#gemm}, in the plan's GEMM mode. */
            void gemm(String task, FloatArray a, TimesFmPacked.Lin l, FloatArray c, int m, int flags) {
                if (gemmMode.cublas()) {
                    cublas(task, a, l, c, m, flags);
                    written(c);
                    return;
                }
                int f = flags | (l.b() >= 0 ? Kernels.BIAS : 0);
                WorkerGrid wg;
                if (gemmMode == GemmMode.SIMDGROUP) {
                    boolean wide = m >= simdgroupWideMinRows;
                    int t = wide ? Kernels.MMA_WIDE : Kernels.MMA_TILE;
                    int tiles = ((m + t - 1) / t) * ((l.out() + t - 1) / t);
                    wg = group(tiles * Kernels.MMA_THREADS, Kernels.MMA_THREADS);
                    int slices = wide ? splitK(tiles, l.in()) : 1;
                    if (slices > 1) {
                        int kSlice = ((l.in() + slices - 1) / slices + Kernels.MMA_K - 1) / Kernels.MMA_K * Kernels.MMA_K;
                        slices = (l.in() + kSlice - 1) / kSlice;
                        FloatArray pa = partials();
                        grid.addWorkerGrid(name + "." + task, group(slices * tiles * Kernels.MMA_THREADS, Kernels.MMA_THREADS));
                        g.task(task, Kernels::gemmSimdgroupWide, ctx, a, 0, w, l.w(), 0, pa, m, l.out(), l.in(), 0, kSlice);
                        items(task + "_sum", m * l.out());
                        g.task(task + "_sum", Kernels::splitKSum, ctx, pa, slices, w, Math.max(l.b(), 0), c, m, l.out(), f);
                        written(c);
                        return;
                    }
                    if (wide) {
                        g.task(task, Kernels::gemmSimdgroupWide, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, m, l.out(), l.in(), f, l.in());
                    } else {
                        g.task(task, Kernels::gemmSimdgroup, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, m, l.out(), l.in(), f);
                    }
                } else if (gemmMode == GemmMode.HALF) {
                    HalfFloatArray ah = half(task, a, m * l.in());
                    HalfFloatArray wh = weights.halfMatrix(l);
                    weight(wh);
                    int t = Kernels.HGEMM_TILE, threads = Kernels.HGEMM_THREADS;
                    wg = new WorkerGrid2D((l.out() + t - 1) / t * threads, (m + t - 1) / t * threads);
                    wg.setLocalWork(threads, threads, 1);
                    g.task(task, Kernels::gemmHalf, ctx, ah, wh, c, w, Math.max(l.b(), 0), m, l.out(), l.in(), f, noTwin, 0);
                } else if (gemmMode == GemmMode.BLOCKED && m >= blockedWideMinRows) {
                    int tr = Kernels.WIDE_ROWS, tc = Kernels.WIDE_COLS, threads = Kernels.HGEMM_THREADS;
                    wg = new WorkerGrid2D((l.out() + tc - 1) / tc * threads, (m + tr - 1) / tr * threads);
                    wg.setLocalWork(threads, threads, 1);
                    g.task(task, Kernels::gemmBlockedWide, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, 0, m, l.out(), l.in(), f);
                } else if (gemmMode == GemmMode.BLOCKED && m >= blockedMinRows) {
                    int t = Kernels.HGEMM_TILE, threads = Kernels.HGEMM_THREADS;
                    wg = new WorkerGrid2D((l.out() + t - 1) / t * threads, (m + t - 1) / t * threads);
                    wg.setLocalWork(threads, threads, 1);
                    g.task(task, Kernels::gemmBlocked, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, 0, m, l.out(), l.in(), f);
                } else {
                    int tile = Kernels.TILE;
                    wg = new WorkerGrid2D((l.out() + tile - 1) / tile * tile, (m + tile - 1) / tile * tile);
                    wg.setLocalWork(tile, tile, 1);
                    g.task(task, Kernels::gemm, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, 0, m, l.out(), l.in(), f);
                }
                grid.addWorkerGrid(name + "." + task, wg);
                written(c);
            }

            /**
             * The GEMM in cuBLAS: row-major {@code C[m, n] = A[m, k]·W[n, k]ᵀ} as the column-major
             * {@code gemm(T, N, n, m, k, W, k, A, k, C, n)}. A bias is written into C first (then β = 1),
             * except before a SiLU, where one epilogue adds the bias and applies the SiLU after the GEMM.
             */
            private void cublas(String task, FloatArray a, TimesFmPacked.Lin l, FloatArray c, int m, int flags) {
                int nOut = l.out(), k = l.in();
                boolean bias = l.b() >= 0;
                boolean acc = (flags & Kernels.ACCUMULATE) != 0;
                boolean silu = (flags & Kernels.SILU) != 0;
                if (silu && acc) {
                    throw new IllegalStateException("GEMM " + task + ": SiLU after accumulation is not supported");
                }
                if (bias && !silu) {
                    items(task + "_bias", m * nOut);
                    g.task(task + "_bias", Kernels::biasFill, ctx, c, w, l.b(), m, nOut, acc ? 1 : 0);
                }
                float beta = (bias && !silu) || acc ? 1f : 0f;
                if (gemmMode == GemmMode.CUBLAS_FP16) {
                    HalfFloatArray ah = half(task, a, m * k);
                    HalfFloatArray wh = weights.halfMatrix(l);
                    weight(wh);
                    g.libraryTask(task, CuBlas::cublasGemmExFP16FP32, 1, 0, nOut, m, k, 1f, wh, k, ah, k, beta, c, nOut);
                } else {
                    FloatArray wm = weights.matrix(l);
                    weight(wm);
                    if (gemmMode == GemmMode.CUBLAS_TF32) {
                        g.libraryTask(task, CuBlas::cublasSgemmTF32, 1, 0, nOut, m, k, 1f, wm, k, a, k, beta, c, nOut);
                    } else {
                        g.libraryTask(task, CuBlas::cublasSgemm, 1, 0, nOut, m, k, 1f, wm, k, a, k, beta, c, nOut);
                    }
                }
                if (silu) {
                    items(task + "_silu", m * nOut);
                    g.task(task + "_silu", Kernels::biasSilu, ctx, c, w, bias ? l.b() : -1, m, nOut, noTwin, 0);
                }
            }

            private HalfFloatArray half(String task, FloatArray a, int count) {
                HalfFloatArray h = halves.get(a);
                if (h == null) {
                    h = new HalfFloatArray(a.getSize());
                    halves.put(a, h);
                    g.transferToDevice(DataTransferMode.FIRST_EXECUTION, h);
                }
                if (fresh.add(a)) {
                    items(task + "_h", count);
                    g.task(task + "_h", Kernels::toHalf, ctx, a, h, count);
                }
                return h;
            }

            private FloatArray partials() {
                if (part == null) {
                    part = new FloatArray(2 * splitKTarget * Kernels.MMA_WIDE * Kernels.MMA_WIDE);
                    g.transferToDevice(DataTransferMode.FIRST_EXECUTION, part);
                }
                return part;
            }

            private void written(FloatArray buf) {
                fresh.remove(buf);
            }

            void rms(String task, FloatArray in, FloatArray out, int wOff, int rows) {
                grid.addWorkerGrid(name + "." + task, group(rows * Kernels.NORM_THREADS, Kernels.NORM_THREADS));
                g.task(task, Kernels::rmsNorm, ctx, in, out, w, wOff, config.hiddenSize(), config.rmsNormEps(), noTwin, 0, rows);
                written(out);
            }

            void items(String task, int count) {
                grid.addWorkerGrid(name + "." + task, group((count + ITEM_GROUP - 1) / ITEM_GROUP * ITEM_GROUP, ITEM_GROUP));
            }

            /** Uploads one run's tokens, padding counts and cache offset. */
            void upload(Batch b, int start) {
                int rows = shape.rows(), p = config.patchLength();
                float[] t = new float[rows * n * 2 * p];
                for (int i = 0; i < rows * n; i++) {
                    System.arraycopy(b.values(), i * p, t, i * 2 * p, p);
                    System.arraycopy(b.masks(), i * p, t, i * 2 * p + p, p);
                }
                MemorySegment.copy(t, 0, tok.getSegment(), ValueLayout.JAVA_FLOAT, 0, t.length);
                if (first) {
                    for (int r = 0; r < rows; r++) {
                        int mk = 0;
                        while (mk < n && b.masks()[((r * n + mk) + 1) * p - 1] > 0.5f) {
                            mk++;
                        }
                        masked.set(r, mk);
                        if (step != null) {
                            step.masked.set(r, mk);
                        }
                    }
                }
                off.set(0, start);
            }

            Heads heads() {
                float[] point = new float[pt.getSize()];
                MemorySegment.copy(pt.getSegment(), ValueLayout.JAVA_FLOAT, 0, point, 0, point.length);
                float[] s = null;
                if (spread) {
                    s = new float[sp.getSize()];
                    MemorySegment.copy(sp.getSegment(), ValueLayout.JAVA_FLOAT, 0, s, 0, s.length);
                }
                return new Heads(point, s);
            }
        }

        @Override
        public long deviceBytes() {
            return exec.getCurrentDeviceMemoryUsage();
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
