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
package ai.beestream.jitcast.timestral.tornado;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitcast.timestral.BlockForward;
import ai.beestream.jitcast.timestral.BlockInput;
import ai.beestream.jitcast.timestral.BlockLayout;
import ai.beestream.jitcast.timestral.BlockPrep;
import ai.beestream.jitcast.timestral.Quantiles;
import ai.beestream.jitcast.timestral.TimestralConfig;
import ai.beestream.jitcast.timestral.Timing;
import ai.beestream.jitcast.timestral.cpu.CpuTimestral;
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
 * {@code TimestralModel.predict_block} on a TornadoVM device: the scaler and the patch tokens, and
 * the quantile head's monotone fan-out and denormalization run on the host ({@link BlockPrep}); the
 * patch embedding, the transformer and the head's projections run as one task graph per block
 * shape (rows, encoder tokens, horizon patches, covariates), compiled once and cached.
 *
 * <p>The device is {@code -Djitcast.tornado.device=<driver>:<device>}, or the default one. GEMMs run
 * as {@code -Dtimestral.tornado.gemm} says ({@link GemmMode}; {@code auto} by default: cuBLAS in
 * FP32 on CUDA, the register-blocked FP32 kernel elsewhere). The blocked kernel takes GEMMs from
 * {@code -Dtimestral.tornado.blockedMinRows} rows (128 by default; its 64-row tiles waste most of
 * their work below that) and the tiled kernel the smaller ones. With cuBLAS the task graph is
 * captured and replayed as a CUDA graph ({@code -Dtimestral.tornado.cudaGraph}, on by default with
 * cuBLAS).
 */
public final class TimestralForward implements BlockForward, AutoCloseable {

    private static final String G = "ts";

    /** Work-group size of the element-wise kernels. */
    private static final int ITEM_GROUP = 128;

    /**
     * Blocks are padded to shared shapes so that few plans are compiled ({@code -Dtimestral.tornado.buckets},
     * on by default): the context with leading empty patches and the rows with empty series of their
     * own, each up to its {@link #bucket}. Neither is ever a valid key, and time positions count from
     * each series' first observation, so the forecasts do not change.
     */
    private final boolean buckets = !"false".equalsIgnoreCase(System.getProperty("timestral.tornado.buckets", "true"));

    /**
     * The size {@code x} is padded to: {@code x} itself up to 8, then the next multiple of a quarter of
     * its power of two (9–16 in steps of 2, 17–32 in steps of 4, …), so padding adds at most 25% and
     * the number of shapes grows with the logarithm of the size.
     */
    static int bucket(int x) {
        if (x <= 8) {
            return x;
        }
        int step = Integer.highestOneBit(x - 1) / 4;
        return (x + step - 1) / step * step;
    }

    /** The fewest rows a GEMM needs to run on {@link Kernels#gemmBlocked}. */
    private final int blockedMinRows = Integer.getInteger("timestral.tornado.blockedMinRows", 128);

    /** The fewest GEMM rows that use {@link Kernels#gemmSimdgroupWide} rather than {@link Kernels#gemmSimdgroup}. */
    private final int wideMinRows = Integer.getInteger("timestral.tornado.simdgroupWideMinRows", 64);

    /**
     * Split K: a 64 × 64 simdgroup GEMM with fewer tiles than {@code splitKTarget} work-groups also
     * splits K, into at most {@code splitKMax} slices of at least {@link #SPLIT_K_MIN} each, so that
     * GEMMs with few rows still fill the GPU. 1 turns it off.
     */
    private final int splitKTarget = Integer.getInteger("timestral.tornado.splitKTarget", 320);
    private final int splitKMax = Math.max(1, Integer.getInteger("timestral.tornado.splitKMax", 16));
    static final int SPLIT_K_MIN = 128;

    /** The number of K slices for a GEMM of {@code tiles} 64 × 64 tiles over {@code k}. */
    int splitK(int tiles, int k) {
        if (tiles >= splitKTarget) {
            return 1;
        }
        return Math.max(1, Math.min(Math.min(splitKMax, (splitKTarget + tiles - 1) / tiles), k / SPLIT_K_MIN));
    }

    /** One compiled block shape. */
    public record Shape(int rows, int encoder, int horizon, boolean covariates) {}

    private final TimestralConfig config;
    private final TimestralPacked weights;
    private final Quantiles quantiles;
    private final float[] invFreq;
    private final int maxCachedPlans;
    /**
     * The most host memory the cached plans' buffers may take together ({@code -Dtimestral.tornado.planBytes},
     * 8 GiB): the least recently used plans are evicted until a new one fits, so a few long-context
     * shapes do not exhaust the JVM's direct memory. A plan larger than the budget still runs, alone.
     */
    private final long maxPlanBytes = Long.getLong("timestral.tornado.planBytes", 8L << 30);
    private final Map<Shape, Plan> plans = new LinkedHashMap<>(16, 0.75f, true);
    private TornadoDevice device;
    private final GemmMode gemmMode;
    /**
     * Time attention as cuBLAS batched GEMMs plus a masked softmax instead of
     * {@link TimestralKernels#timeAttention} ({@code -Dtimestral.tornado.attention}): {@code auto} with a
     * cuBLAS GEMM mode for blocks of at least {@code blasAttentionMinTokens} tokens per row (128) and at
     * most {@code blasAttentionMaxRows} rows (8): long rows, where the kernel's one-key-at-a-time loop
     * is slowest, but few of them, since the batched GEMMs compute and move every score, masked or not,
     * and lose to the kernel on large batches; {@code cublas} always (with a cuBLAS GEMM mode);
     * {@code kernel} never. The scores of one chunk of (row, head) batches take at most
     * {@code -Dtimestral.tornado.attentionScoreBytes} (512 MB).
     */
    private final String attentionMode = System.getProperty("timestral.tornado.attention", "auto").toLowerCase();
    private final int blasAttentionMinTokens = Integer.getInteger("timestral.tornado.blasAttentionMinTokens", 128);
    private final int blasAttentionMaxRows = Integer.getInteger("timestral.tornado.blasAttentionMaxRows", 8);
    private final long attentionScoreBytes = Long.getLong("timestral.tornado.attentionScoreBytes", 512L << 20);
    private final boolean cudaGraph;

    public TimestralForward(TimestralConfig config, TimestralPacked weights) {
        this(config, weights, Integer.getInteger("timestral.tornado.plans", 8));
    }

    public TimestralForward(TimestralConfig config, TimestralPacked weights, int maxCachedPlans) {
        if (config.headDim() > TimestralKernels.MAX_HEAD_DIM || config.headDim() % (TimestralKernels.ATTN_THREADS
                / TimestralKernels.ATTN_QUERIES) != 0) {
            throw new IllegalArgumentException("unsupported attention head width " + config.headDim());
        }
        this.config = config;
        this.weights = weights;
        this.quantiles = new Quantiles(config);
        this.invFreq = CpuTimestral.inverseFrequencies(config);
        this.maxCachedPlans = Math.max(1, maxCachedPlans);
        this.gemmMode = GemmMode.parse(System.getProperty("timestral.tornado.gemm", "auto")).resolve(device());
        this.cudaGraph = Boolean.parseBoolean(System.getProperty("timestral.tornado.cudaGraph", String.valueOf(gemmMode.cublas())));
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
    public float[][][] predictBlock(BlockInput block) {
        if (buckets) {
            BlockInput padded = bucketed(block, config.patchSize());
            if (padded != block) {
                return java.util.Arrays.copyOf(predictExact(padded), block.rows());
            }
        }
        return predictExact(block);
    }

    /** {@code block} padded to the shared shapes, or {@code block} itself when it already has one. */
    static BlockInput bucketed(BlockInput b, int ps) {
        int r = b.rows(), t = b.length();
        int patches = t / ps;
        int padPatches = bucket(patches);
        int rows = bucket(r);
        if (padPatches == patches && rows == r) {
            return b;
        }
        int lead = (padPatches - patches) * ps, len = padPatches * ps;
        float[][] inputs = new float[rows][len];
        boolean[][] observed = new boolean[rows][len];
        int[] group = new int[rows];
        int[] role = b.role() == null ? null : new int[rows];
        int fl = b.knownFuture() == null ? 0 : b.knownFuture()[0].length;
        float[][] kf = b.knownFuture() == null ? null : new float[rows][fl];
        boolean[][] kfm = b.knownFutureMask() == null ? null : new boolean[rows][fl];
        int nextGroup = java.util.Arrays.stream(b.group()).max().orElse(0) + 1;
        for (int i = 0; i < rows; i++) {
            if (i < r) {
                System.arraycopy(b.inputs()[i], 0, inputs[i], lead, t);
                System.arraycopy(b.observed()[i], 0, observed[i], lead, t);
                group[i] = b.group()[i];
                if (role != null) {
                    role[i] = b.role()[i];
                }
                if (kf != null) {
                    System.arraycopy(b.knownFuture()[i], 0, kf[i], 0, fl);
                    System.arraycopy(b.knownFutureMask()[i], 0, kfm[i], 0, fl);
                }
            } else {
                group[i] = nextGroup++; // an empty series of its own (role 0, a target)
            }
        }
        return new BlockInput(inputs, observed, group, b.horizonPatches(), role, kf, kfm);
    }

    private float[][][] predictExact(BlockInput block) {
        long t = Timing.start();
        BlockLayout lay = BlockLayout.of(config, block);
        Timing.add("fwd.layout", t);
        t = Timing.start();
        BlockPrep prep = BlockPrep.of(config, lay);
        Timing.add("fwd.prep", t);
        Shape shape = new Shape(lay.rows(), lay.encoderTokens(), lay.horizonPatches(), lay.covariates());
        Plan plan = plans.get(shape);
        if (plan == null) {
            // evict before building, so the old plan's device buffers are freed first
            long need = planBytes(shape);
            while (!plans.isEmpty() && (plans.size() >= maxCachedPlans || cachedBytes() + need > maxPlanBytes)) {
                Shape eldest = plans.keySet().iterator().next();
                plans.remove(eldest).close();
            }
            long tb = Timing.start();
            plan = new Plan(shape);
            plans.put(shape, plan);
            Timing.add("fwd.plan-build", tb);
        }
        float[] raw = plan.run(lay, prep);
        t = Timing.start();
        float[][][] out = prep.finish(config, lay, quantiles, raw);
        Timing.add("fwd.finish", t);
        return out;
    }

    private long cachedBytes() {
        long sum = 0;
        for (Shape s : plans.keySet()) {
            sum += planBytes(s);
        }
        return sum;
    }

    /** Whether a block shape runs its time attention as cuBLAS batched GEMMs. */
    boolean blasAttention(Shape shape) {
        if (!gemmMode.cublas() || attentionMode.equals("kernel")) {
            return false;
        }
        return attentionMode.equals("cublas")
                || shape.encoder() + shape.horizon() >= blasAttentionMinTokens && shape.rows() <= blasAttentionMaxRows;
    }

    /** The (row, head) batches per chunk of the cuBLAS time attention. */
    int attentionChunk(Shape shape) {
        long n = shape.encoder() + shape.horizon();
        return (int) Math.max(1, Math.min((long) shape.rows() * config.numHeads(), attentionScoreBytes / (4 * n * n)));
    }

    /** The host memory of one plan's buffers (those {@link Plan} allocates for its shape). */
    long planBytes(Shape shape) {
        TimestralConfig c = config;
        long rows = shape.rows(), enc = shape.encoder(), h = shape.horizon(), n = enc + h;
        long d = c.dModel(), ps = c.patchSize(), f = c.mlpHiddenDim(), q = c.numQuantiles();
        long tokens = rows * n, encTokens = rows * enc, decTokens = rows * h;
        long floats = encTokens * 2 * ps + encTokens * d + 5 * tokens * d + tokens * Math.max(Math.max(3 * d, 2 * f), 4 * d)
                + tokens * f + decTokens * d + decTokens * ps * q + (2 * enc + h) * c.headDim();
        long ints = tokens + encTokens + 4 * rows;
        if (gemmMode == GemmMode.SIMDGROUP) {
            floats += 2L * splitKTarget * Kernels.MMA_WIDE * Kernels.MMA_WIDE;
        }
        if (blasAttention(shape)) {
            long chunk = attentionChunk(shape);
            floats += chunk * n * n + 4 * chunk * n * c.headDim();
        }
        return 4 * (floats + ints);
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
        final FloatArray tok, emb, x, hn, big, att, gate, act, y, dec, raw, table;
        final IntArray pos, keyValid, known, groupStart, groupEnd, role;
        final HalfFloatArray noTwin = new HalfFloatArray(1);
        final TornadoExecutionPlan exec;
        final GridScheduler grid = new GridScheduler();
        final KernelContext ctx = new KernelContext();
        final FloatArray w = weights.data();
        TaskGraph g;
        FloatArray part;
        /** FP16 copies of GEMM inputs, and the ones whose copy is current (no task has written the source since). */
        final Map<FloatArray, HalfFloatArray> halves = new IdentityHashMap<>();
        final Set<FloatArray> fresh = Collections.newSetFromMap(new IdentityHashMap<>());
        /** The per-matrix weight copies already added to the graph. */
        final Set<Object> graphWeights = Collections.newSetFromMap(new IdentityHashMap<>());

        Plan(Shape shape) {
            this.shape = shape;
            TimestralConfig c = config;
            int rows = shape.rows(), enc = shape.encoder(), h = shape.horizon(), n = enc + h;
            int d = c.dModel(), ps = c.patchSize(), f = c.mlpHiddenDim(), q = c.numQuantiles();
            int tokens = rows * n, encTokens = rows * enc, decTokens = rows * h;
            tok = new FloatArray(encTokens * 2 * ps);
            emb = new FloatArray(encTokens * d);
            x = new FloatArray(tokens * d);
            hn = new FloatArray(tokens * d);
            big = new FloatArray(tokens * Math.max(Math.max(3 * d, 2 * f), 4 * d));
            att = new FloatArray(tokens * d);
            gate = new FloatArray(tokens * d);
            act = new FloatArray(tokens * f);
            y = new FloatArray(tokens * d);
            dec = new FloatArray(decTokens * d);
            raw = new FloatArray(decTokens * ps * q);
            table = new FloatArray((2 * enc + h) * c.headDim());
            pos = new IntArray(tokens);
            keyValid = new IntArray(encTokens);
            known = new IntArray(rows);
            groupStart = new IntArray(rows);
            groupEnd = new IntArray(rows);
            role = new IntArray(rows);

            g = new TaskGraph(G);
            g.transferToDevice(DataTransferMode.FIRST_EXECUTION, w, emb, x, hn, big, att, gate, act, y, dec, raw, noTwin);
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, tok, table, pos, keyValid, known, groupStart, groupEnd, role);

            residual("emb", tok, big, emb, encTokens, weights.embedding());
            items("assemble", tokens * d);
            g.task("assemble", TimestralKernels::assembleTokens, ctx, emb, w, weights.forecastQuery(), x, rows, enc, n, d);
            written(x);

            int heads = c.numHeads(), dh = c.headDim();
            int qTiles = (n + TimestralKernels.ATTN_QUERIES - 1) / TimestralKernels.ATTN_QUERIES;
            int rTiles = (rows + TimestralKernels.ATTN_QUERIES - 1) / TimestralKernels.ATTN_QUERIES;
            for (int li = 0; li < weights.layers().size(); li++) {
                TimestralPacked.Layer layer = weights.layers().get(li);
                for (int ai = 0; ai < layer.attentions().size(); ai++) {
                    TimestralPacked.Attention a = layer.attentions().get(ai);
                    String s = "l" + li + "a" + ai + "_";
                    rms(s + "norm", x, hn, a.norm(), tokens, c.normEps());
                    gemm(s + "qkv", hn, a.qkv(), big, tokens, 0);
                    if (a.time()) {
                        items(s + "rope", tokens * heads * (dh / 2));
                        g.task(s + "rope", TimestralKernels::rope, ctx, big, pos, table, tokens, heads, dh, d);
                        written(big);
                        if (blasAttention(shape)) {
                            timeAttentionBlas(s, rows, enc, enc - (shape.covariates() ? h : 0), n, heads, dh, d);
                        } else {
                            grid.addWorkerGrid(G + "." + s + "attn", group(rows * heads * qTiles * TimestralKernels.ATTN_THREADS,
                                    TimestralKernels.ATTN_THREADS));
                            g.task(s + "attn", TimestralKernels::timeAttention, ctx, big, att, keyValid, known, rows, enc,
                                    enc - (shape.covariates() ? h : 0), n, heads, dh, d);
                        }
                    } else {
                        grid.addWorkerGrid(G + "." + s + "attn", group(n * heads * rTiles * TimestralKernels.ATTN_THREADS,
                                TimestralKernels.ATTN_THREADS));
                        g.task(s + "attn", TimestralKernels::variateAttention, ctx, big, att, keyValid, groupStart, groupEnd,
                                role, shape.covariates() ? 1 : 0, rows, enc, n, heads, dh, d);
                    }
                    written(att);
                    if (a.gate() != null) {
                        gemm(s + "gproj", hn, a.gate(), gate, tokens, 0);
                        items(s + "gate", tokens * d);
                        g.task(s + "gate", TimestralKernels::gate, ctx, att, gate, tokens * d);
                        written(att);
                    }
                    gemm(s + "wo", att, a.wo(), x, tokens, Kernels.ACCUMULATE);
                }
                String s = "l" + li + "_";
                rms(s + "mnorm", x, hn, layer.mlpNorm(), tokens, c.normEps());
                gemm(s + "up", hn, layer.up(), big, tokens, 0);
                items(s + "swiglu", tokens * f);
                g.task(s + "swiglu", Kernels::swiGlu, ctx, big, act, tokens, f, noTwin, 0);
                written(act);
                gemm(s + "down", act, layer.down(), x, tokens, Kernels.ACCUMULATE);
            }
            rms("onorm", x, y, weights.outNorm(), tokens, c.normEps());
            items("gather", decTokens * d);
            g.task("gather", TimestralKernels::gatherDecoder, ctx, y, dec, rows, enc, n, h, d);
            written(dec);
            residual("head", dec, big, raw, decTokens, weights.head());
            g.transferToHost(DataTransferMode.EVERY_EXECUTION, raw);

            exec = new TornadoExecutionPlan(g.snapshot());
            exec.withGridScheduler(grid);
            exec.withDevice(device());
            if (cudaGraph) {
                // CUDA backend only: the task graph is captured once and replayed as a CUDA graph
                exec.withCUDAGraph();
            }
            if (Boolean.getBoolean("timestral.tornado.profile")) {
                exec.withProfiler(uk.ac.manchester.tornado.api.enums.ProfilerMode.CONSOLE);
            }
        }

        /** {@code output(SiLU(hidden(a))) + residual(a)}, with the hidden activation in {@code scratch}. */
        void residual(String name, FloatArray a, FloatArray scratch, FloatArray out, int m, TimestralPacked.Residual r) {
            gemm(name + "_h", a, r.hidden(), scratch, m, Kernels.SILU);
            gemm(name + "_o", scratch, r.output(), out, m, 0);
            gemm(name + "_r", a, r.residual(), out, m, Kernels.ACCUMULATE);
        }

        /** {@code C = act(A·Wᵀ + b) (+ C)}, with the flags of {@link Kernels#gemm}, in the plan's GEMM mode. */
        void gemm(String name, FloatArray a, TimestralPacked.Lin l, FloatArray c, int m, int flags) {
            if (gemmMode.cublas()) {
                cublas(name, a, l, c, m, flags);
                written(c);
                return;
            }
            int f = flags | (l.b() >= 0 ? Kernels.BIAS : 0);
            WorkerGrid wg;
            if (gemmMode == GemmMode.SIMDGROUP) {
                boolean wide = m >= wideMinRows;
                int t = wide ? Kernels.MMA_WIDE : Kernels.MMA_TILE;
                int tiles = ((m + t - 1) / t) * ((l.out() + t - 1) / t);
                wg = group(tiles * Kernels.MMA_THREADS, Kernels.MMA_THREADS);
                int slices = wide ? splitK(tiles, l.in()) : 1;
                if (slices > 1) {
                    int kSlice = ((l.in() + slices - 1) / slices + Kernels.MMA_K - 1) / Kernels.MMA_K * Kernels.MMA_K;
                    slices = (l.in() + kSlice - 1) / kSlice;
                    FloatArray p = partials();
                    grid.addWorkerGrid(G + "." + name, group(slices * tiles * Kernels.MMA_THREADS, Kernels.MMA_THREADS));
                    g.task(name, Kernels::gemmSimdgroupWide, ctx, a, 0, w, l.w(), 0, p, m, l.out(), l.in(), 0, kSlice);
                    items(name + "_sum", m * l.out());
                    g.task(name + "_sum", Kernels::splitKSum, ctx, p, slices, w, Math.max(l.b(), 0), c, m, l.out(), f);
                    written(c);
                    return;
                }
                if (wide) {
                    g.task(name, Kernels::gemmSimdgroupWide, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, m, l.out(), l.in(), f, l.in());
                } else {
                    g.task(name, Kernels::gemmSimdgroup, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, m, l.out(), l.in(), f);
                }
            } else if (gemmMode == GemmMode.HALF) {
                HalfFloatArray ah = half(name, a, m * l.in());
                HalfFloatArray wh = halfWeight(l);
                int t = Kernels.HGEMM_TILE, threads = Kernels.HGEMM_THREADS;
                wg = new WorkerGrid2D((l.out() + t - 1) / t * threads, (m + t - 1) / t * threads);
                wg.setLocalWork(threads, threads, 1);
                g.task(name, Kernels::gemmHalf, ctx, ah, wh, c, w, Math.max(l.b(), 0), m, l.out(), l.in(), f, noTwin, 0);
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
         * into C first (then β = 1), except before a SiLU, where one epilogue adds the bias and
         * applies the SiLU after the GEMM.
         */
        private void cublas(String name, FloatArray a, TimestralPacked.Lin l, FloatArray c, int m, int flags) {
            int n = l.out(), k = l.in();
            boolean bias = l.b() >= 0;
            boolean acc = (flags & Kernels.ACCUMULATE) != 0;
            boolean silu = (flags & Kernels.SILU) != 0;
            if (silu && acc) {
                throw new IllegalStateException("GEMM " + name + ": SiLU after accumulation is not supported");
            }
            if (bias && !silu) {
                items(name + "_bias", m * n);
                g.task(name + "_bias", Kernels::biasFill, ctx, c, w, l.b(), m, n, acc ? 1 : 0);
            }
            float beta = (bias && !silu) || acc ? 1f : 0f;
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
            if (silu) {
                items(name + "_silu", m * n);
                g.task(name + "_silu", Kernels::biasSilu, ctx, c, w, bias ? l.b() : -1, m, n, noTwin, 0);
            }
        }

        /** Heads-major q, k, v and output, and the scores, of one chunk of {@link #timeAttentionBlas}. */
        FloatArray qh, kh, vh, oh, scores;

        /**
         * Time attention as cuBLAS batched GEMMs, FP32: per chunk of batches (row, head), q, k and v are
         * copied heads-major, {@code S = q·kᵀ·scale} is one strided-batched SGEMM, the masked softmax
         * runs in place on S, and {@code o = S·v} is a second one. Chunks keep S within
         * {@code attentionScoreBytes}.
         */
        private void timeAttentionBlas(String s, int rows, int enc, int ll, int n, int heads, int dh, int d) {
            int batches = rows * heads;
            int chunk = attentionChunk(shape);
            if (scores == null) {
                qh = new FloatArray(chunk * n * dh);
                kh = new FloatArray(chunk * n * dh);
                vh = new FloatArray(chunk * n * dh);
                oh = new FloatArray(chunk * n * dh);
                scores = new FloatArray(chunk * n * n);
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, qh, kh, vh, oh, scores);
            }
            float scale = (float) (1.0 / Math.sqrt(dh));
            long nd = (long) n * dh, nn = (long) n * n;
            for (int b0 = 0, c = 0; b0 < batches; b0 += chunk, c++) {
                String p = s + "attn" + c + "_";
                items(p + "split", chunk * n * dh);
                g.task(p + "split", TimestralKernels::splitHeads, ctx, big, qh, kh, vh, b0, chunk, rows, n, heads, dh, d);
                // column-major: S_cm[key, query] = (k_cm)ᵀ·q_cm, i.e. row-major S[query][key]
                g.libraryTask(p + "qk", CuBlas::cublasSgemmStridedBatched, 1, 0, n, n, dh, scale, kh, dh, nd, qh, dh, nd, 0f, scores,
                        n, nn, chunk);
                grid.addWorkerGrid(G + "." + p + "softmax", group(chunk * n * TimestralKernels.SOFTMAX_THREADS,
                        TimestralKernels.SOFTMAX_THREADS));
                g.task(p + "softmax", TimestralKernels::attentionSoftmax, ctx, scores, keyValid, known, b0, chunk, rows, enc, ll, n,
                        heads);
                // column-major: o_cm[j, query] = v_cm·S_cm, i.e. row-major o[query][j] = Σ_key S[query][key]·v[key][j]
                g.libraryTask(p + "pv", CuBlas::cublasSgemmStridedBatched, 0, 0, dh, n, n, 1f, vh, dh, nd, scores, n, nn, 0f, oh, dh,
                        nd, chunk);
                items(p + "merge", chunk * n * dh);
                g.task(p + "merge", TimestralKernels::mergeHeads, ctx, oh, att, b0, chunk, rows, n, heads, dh, d);
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

        private HalfFloatArray halfWeight(TimestralPacked.Lin l) {
            HalfFloatArray wh = weights.halfMatrix(l);
            if (graphWeights.add(wh)) {
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, wh);
            }
            return wh;
        }

        /**
         * The split-K partial tiles, shared by every split GEMM of the plan (they run one after another).
         * A split GEMM has fewer tiles than the target and at most ⌈target / tiles⌉ slices, so its
         * partials take fewer than two targets' worth of tiles.
         */
        private FloatArray partials() {
            if (part == null) {
                part = new FloatArray(2 * splitKTarget * Kernels.MMA_WIDE * Kernels.MMA_WIDE);
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, part);
            }
            return part;
        }

        /** A task has written {@code buf}: its FP16 copy is stale. */
        private void written(FloatArray buf) {
            fresh.remove(buf);
        }

        void rms(String name, FloatArray in, FloatArray out, int wOff, int rows, float eps) {
            grid.addWorkerGrid(G + "." + name, group(rows * Kernels.NORM_THREADS, Kernels.NORM_THREADS));
            g.task(name, Kernels::rmsNorm, ctx, in, out, w, wOff, config.dModel(), eps, noTwin, 0);
            written(out);
        }

        void items(String task, int count) {
            grid.addWorkerGrid(G + "." + task, group((count + ITEM_GROUP - 1) / ITEM_GROUP * ITEM_GROUP, ITEM_GROUP));
        }

        /** Uploads one block's per-call inputs, runs the graph, returns the head's raw outputs. */
        float[] run(BlockLayout lay, BlockPrep prep) {
            int rows = shape.rows(), enc = shape.encoder(), h = shape.horizon(), n = enc + h, dh = config.headDim();
            long t0 = Timing.start();
            float[] t = prep.tokens();
            MemorySegment.copy(t, 0, tok.getSegment(), ValueLayout.JAVA_FLOAT, 0, t.length);
            // RoPE table over every position a token can have: (−enc, enc + h)
            int half = dh / 2;
            for (int p = 0; p < 2 * enc + h; p++) {
                float position = p - enc;
                for (int e = 0; e < half; e++) {
                    float a = position * invFreq[e];
                    table.set(p * dh + e, (float) Math.cos(a));
                    table.set(p * dh + half + e, (float) Math.sin(a));
                }
            }
            Set<Integer> seen = new HashSet<>();
            for (int i = 0; i < rows; i++) {
                for (int tk = 0; tk < n; tk++) {
                    pos.set(i * n + tk, lay.position(i, tk) + enc);
                }
                for (int p = 0; p < enc; p++) {
                    keyValid.set(i * enc + p, lay.keyValid(i, p) ? 1 : 0);
                }
                known.set(i, lay.known(i) ? 1 : 0);
                role.set(i, lay.role(i));
                if (i == 0 || lay.group(i) != lay.group(i - 1)) {
                    if (!seen.add(lay.group(i))) {
                        throw new IllegalArgumentException("rows of series " + lay.group(i) + " are not contiguous");
                    }
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
            Timing.add("fwd.upload", t0);
            t0 = Timing.start();
            exec.execute();
            Timing.add("fwd.device", t0);
            t0 = Timing.start();
            float[] out = new float[raw.getSize()];
            MemorySegment.copy(raw.getSegment(), ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
            Timing.add("fwd.download", t0);
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
