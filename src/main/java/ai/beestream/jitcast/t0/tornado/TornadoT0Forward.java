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
package ai.beestream.jitcast.t0.tornado;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitcast.t0.T0Config;
import ai.beestream.jitcast.t0.T0Forward;
import ai.beestream.jitcast.tornado.Devices;
import ai.beestream.jitcast.tornado.GemmMode;
import ai.beestream.jitcast.tornado.Kernels;
import ai.beestream.jitcast.tornado.PlanCache;
import ai.beestream.jitcast.tornado.SharedWeights;
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
 * t0's forward pass on a TornadoVM device: the patch encoder, the transformer layers (attention across time
 * and across variates), the final norm, and the decoder and quantile head of the patches whose predictions
 * are wanted, as one task graph per shape (rows, patches, output patches), compiled once and cached. The host
 * builds each patch's {@code [values ‖ time index ‖ validity]}, its type and validity, and each row's series.
 *
 * <p>The device is {@code -Djitcast.tornado.device=<backend>:<device>}, or the default one. GEMMs run as
 * {@code -Dt0.tornado.gemm} says ({@link GemmMode}; {@code auto} by default: cuBLAS in FP32 on CUDA, Apple's
 * simdgroup matrices on Metal, the register-blocked FP32 kernel elsewhere). With cuBLAS the task graph is
 * captured and replayed as a CUDA graph ({@code -Dt0.tornado.cudaGraph}, on by default with cuBLAS).
 */
public final class TornadoT0Forward implements T0Forward {

    static {
        Devices.defaults();
    }

    private static final String G = "t0";

    /** Work-group size of the element-wise kernels. */
    private static final int ITEM_GROUP = 128;

    /**
     * A {@link #gemm} flag for the cuBLAS modes: the GEMM runs without its bias, which the kernel reading
     * its output adds instead of a separate pass over it.
     */
    private static final int NO_BIAS = 1 << 16;

    /** The fewest rows a GEMM needs to run on {@link Kernels#gemmBlocked}. */
    private final int blockedMinRows = Options.integer("t0.tornado.blockedMinRows", 128, 1);

    /** The fewest rows a GEMM needs to run on {@link Kernels#gemmBlockedWide}. */
    private final int blockedWideMinRows = Options.integer("t0.tornado.wideMinRows", 1024, 1);

    /** The fewest GEMM rows that use {@link Kernels#gemmSimdgroupWide} rather than {@link Kernels#gemmSimdgroup}. */
    private final int simdgroupWideMinRows = Options.integer("t0.tornado.simdgroupWideMinRows", 64, 1);

    /**
     * Split K: a 64 × 64 simdgroup GEMM with fewer tiles than {@code splitKTarget} work-groups also splits K,
     * into at most {@code splitKMax} slices of at least {@link #SPLIT_K_MIN} each. 1 turns it off.
     */
    private final int splitKTarget = Options.integer("t0.tornado.splitKTarget", 320, 1);
    private final int splitKMax = Options.integer("t0.tornado.splitKMax", 16, 1);
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
     * @param rows the rows (variates of every series)
     * @param patches the patches per row
     * @param count the patches whose predictions are wanted, the last {@code count + 1} but one
     */
    public record Shape(int rows, int patches, int count) {}

    private final T0Config config;
    private final T0Packed weights;
    private final PlanCache<Shape, Plan> plans;
    private final TornadoDevice device = Devices.device();
    /** The one device copy of the weights every plan consumes. */
    private final SharedWeights shared = new SharedWeights(G, device);
    private final GemmMode gemmMode;
    private final boolean cudaGraph;

    /**
     * A forward pass with the default plan count ({@code -Dt0.tornado.plans}, 8).
     *
     * @param config the checkpoint's configuration
     * @param weights the packed weights
     */
    public TornadoT0Forward(T0Config config, T0Packed weights) {
        this(config, weights, Options.integer("t0.tornado.plans", 8, 1));
    }

    /**
     * A forward pass keeping at most {@code maxCachedPlans} compiled shapes.
     *
     * @param config the checkpoint's configuration
     * @param weights the packed weights
     * @param maxCachedPlans the most compiled shapes to keep
     */
    public TornadoT0Forward(T0Config config, T0Packed weights, int maxCachedPlans) {
        if (config.headDim() > T0Kernels.HEAD_DIM) {
            throw new IllegalArgumentException("attention heads wider than " + T0Kernels.HEAD_DIM + " are not supported");
        }
        this.config = config;
        this.weights = weights;
        this.gemmMode = Options.enumeration("t0.tornado.gemm", GemmMode.AUTO, GemmMode.class).resolve(device);
        long weightBytes = SharedWeights.estimate(weights.data().getNumBytesOfSegment(),
                gemmMode == GemmMode.CUBLAS || gemmMode == GemmMode.CUBLAS_TF32, gemmMode == GemmMode.CUBLAS_FP16 || gemmMode == GemmMode.HALF);
        this.plans = new PlanCache<>(Math.max(1, maxCachedPlans),
                Options.bytes("t0.tornado.planBytes", Math.max(1, Devices.memory() - weightBytes), 1));
        this.cudaGraph = Options.bool("t0.tornado.cudaGraph", gemmMode.cublas());
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
    public float[] forward(Input in, int from, int count) {
        if (from != in.patches() - count - 1) {
            throw new IllegalArgumentException("the device forward pass predicts from the last context patch on");
        }
        Shape shape = new Shape(in.rows(), in.patches(), count);
        return plans.run(shape, Plan::new, plan -> plan.run(in));
    }

    @Override
    public void close() {
        try {
            plans.close();
        } finally {
            shared.close();
        }
    }

    private static WorkerGrid group(int global, int local) {
        WorkerGrid wg = new WorkerGrid1D(global);
        wg.setLocalWork(local, 1, 1);
        return wg;
    }

    /** The task graph and execution plan for one shape. */
    private final class Plan implements PlanCache.Plan {
        final Shape shape;
        final FloatArray w = weights.data();
        final GridScheduler grid = new GridScheduler();
        final KernelContext ctx = new KernelContext();
        /** The weight buffers the plan's graph reads, consumed from {@link #shared}. */
        final Set<Object> planWeights = Collections.newSetFromMap(new IdentityHashMap<>());
        final Set<Object> declared = Collections.newSetFromMap(new IdentityHashMap<>());
        final String name;
        final TaskGraph g;
        final TornadoExecutionPlan exec;
        final FloatArray tok, x, hn, big, att, act, dec, hb, raw, out;
        final IntArray type, valid, future, groupOf, groupStart, groupRows;
        final HalfFloatArray noTwin = new HalfFloatArray(1);
        FloatArray part;
        final Map<FloatArray, HalfFloatArray> halves = new IdentityHashMap<>();
        final Set<FloatArray> fresh = Collections.newSetFromMap(new IdentityHashMap<>());

        Plan(Shape shape) {
            this.shape = shape;
            T0Config c = config;
            int rows = shape.rows(), n = shape.patches(), count = shape.count();
            int d = c.embedDim(), ff = c.mlpHiddenDim(), p = c.patchSize(), heads = c.numHeads(), hd = c.headDim();
            int q = c.quantiles();
            long tokensL = (long) rows * n;
            if (tokensL * Math.max(3 * d, 2 * ff) > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("a forward pass of " + tokensL + " patches is too large; use smaller batches");
            }
            int tokens = (int) tokensL, outs = rows * count;
            tok = new FloatArray(tokens * 3 * p);
            x = new FloatArray(tokens * d);
            hn = new FloatArray(tokens * d);
            big = new FloatArray(tokens * Math.max(3 * d, 2 * ff));
            att = new FloatArray(tokens * d);
            act = new FloatArray(tokens * ff);
            dec = new FloatArray(outs * d);
            hb = new FloatArray(outs * d);
            raw = new FloatArray(outs * p * q);
            out = new FloatArray(outs * p * q);
            type = new IntArray(tokens);
            valid = new IntArray(tokens);
            future = new IntArray(rows);
            groupOf = new IntArray(rows);
            groupStart = new IntArray(rows + 1);
            groupRows = new IntArray(rows);

            name = G + "_" + rows + "_" + n + "_" + count;
            g = new TaskGraph(name);
            weight(w);
            g.transferToDevice(DataTransferMode.FIRST_EXECUTION, x, hn, big, att, act, dec, hb, raw, out, noTwin);
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, tok, type, valid, future, groupOf, groupStart, groupRows);

            residual("enc", tok, hn, x, tokens, weights.encoder());
            items("type", tokens * d);
            g.task("type", T0Kernels::addType, ctx, x, w, weights.typeEmbeddings(), type, tokens, d);
            written(x);
            float scale = (float) (1.0 / Math.sqrt(hd));
            // In the cuBLAS modes the layers' GEMMs run without their biases, which the next kernel adds:
            // qkNormRope (q|k|v), biasSwiGlu (up) and biasRmsNorm (out and down, into the residual stream).
            boolean fold = gemmMode.cublas();
            int nb = fold ? NO_BIAS : 0;
            int pending = -1;
            for (int li = 0; li < weights.layers().size(); li++) {
                T0Packed.Layer l = weights.layers().get(li);
                String s = "l" + li + "_";
                norm(s + "an", x, hn, pending, l.attnNorm(), tokens);
                gemm(s + "qkv", hn, l.qkv(), big, tokens, nb);
                grid.addWorkerGrid(name + "." + s + "qk", group(tokens * heads * T0Kernels.QK_THREADS, T0Kernels.QK_THREADS));
                g.task(s + "qk", T0Kernels::qkNormRope, ctx, big, w, fold ? l.qkv().b() : -1, l.qNorm(), l.kNorm(), weights.invFreq(),
                        weights.logXpos(), l.group() ? 0 : 1, c.xposHalves() ? 1 : 0, rows, n, heads, hd, d, T0Config.NORM_EPS);
                written(big);
                if (l.group()) {
                    grid.addWorkerGrid(name + "." + s + "attn",
                            group(tokens * heads * T0Kernels.GROUP_THREADS, T0Kernels.GROUP_THREADS));
                    g.task(s + "attn", T0Kernels::groupAttention, ctx, big, att, valid, future, groupOf, groupStart, groupRows, rows, n,
                            heads, hd, d, scale);
                } else {
                    int tiles = (n + T0Kernels.ATTN_QUERIES - 1) / T0Kernels.ATTN_QUERIES;
                    grid.addWorkerGrid(name + "." + s + "attn",
                            group(rows * heads * tiles * T0Kernels.ATTN_THREADS, T0Kernels.ATTN_THREADS));
                    g.task(s + "attn", T0Kernels::timeAttention, ctx, big, att, valid, future, rows, n, heads, hd, d, scale);
                }
                written(att);
                gemm(s + "o", att, l.out(), x, tokens, Kernels.ACCUMULATE | nb);
                norm(s + "mn", x, hn, fold ? l.out().b() : -1, l.mlpNorm(), tokens);
                gemm(s + "up", hn, l.up(), big, tokens, nb);
                items(s + "glu", tokens * ff);
                if (fold) {
                    g.task(s + "glu", T0Kernels::biasSwiGlu, ctx, big, act, w, l.up().b(), tokens, ff);
                } else {
                    g.task(s + "glu", Kernels::swiGlu, ctx, big, act, tokens, ff, noTwin, 0);
                }
                written(act);
                gemm(s + "down", act, l.down(), x, tokens, Kernels.ACCUMULATE | nb);
                pending = fold ? l.down().b() : -1;
            }
            norm("on", x, hn, pending, weights.outNorm(), tokens);
            items("gather", outs * d);
            g.task("gather", T0Kernels::gatherPatches, ctx, hn, dec, rows, n, n - count - 1, count, d);
            written(dec);
            residual("dec", dec, hb, raw, outs, weights.decoder());
            items("head", outs * p);
            g.task("head", T0Kernels::quantileHead, ctx, raw, out, outs * p, q);
            g.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            shared.consume(g, new ArrayList<>(declared));

            exec = new TornadoExecutionPlan(g.snapshot());
            exec.withGridScheduler(grid);
            exec.withDevice(device);
            if (cudaGraph) {
                exec.withCUDAGraph();
            }
            if (Options.bool("t0.tornado.profile", false)) {
                exec.withProfiler(uk.ac.manchester.tornado.api.enums.ProfilerMode.CONSOLE);
            }
        }

        float[] run(Input in) {
            int rows = shape.rows(), n = shape.patches(), p = config.patchSize(), tokens = rows * n;
            float[] t = new float[tokens * 3 * p];
            for (int i = 0; i < tokens; i++) {
                System.arraycopy(in.values(), i * p, t, i * 3 * p, p);
                for (int j = 0; j < p; j++) {
                    t[i * 3 * p + p + j] = (float) j / p;
                }
                System.arraycopy(in.valid(), i * p, t, i * 3 * p + 2 * p, p);
            }
            MemorySegment.copy(t, 0, tok.getSegment(), ValueLayout.JAVA_FLOAT, 0, t.length);
            for (int i = 0; i < tokens; i++) {
                type.set(i, in.patchType()[i]);
                valid.set(i, in.patchValid()[i] ? 1 : 0);
            }
            for (int r = 0; r < rows; r++) {
                future.set(r, in.future()[r] ? 1 : 0);
                groupOf.set(r, in.groupOf()[r]);
                groupRows.set(r, in.groupRows()[r]);
            }
            for (int i = 0; i <= rows; i++) {
                groupStart.set(i, i < in.groupStart().length ? in.groupStart()[i] : rows);
            }
            exec.execute();
            float[] result = new float[out.getSize()];
            MemorySegment.copy(out.getSegment(), ValueLayout.JAVA_FLOAT, 0, result, 0, result.length);
            return result;
        }

        /** A weight buffer the graph reads, consumed from the shared copy once the graph is built. */
        private void weight(Object buf) {
            if (declared.add(buf)) {
                planWeights.add(buf);
            }
        }

        void residual(String task, FloatArray a, FloatArray scratch, FloatArray o, int m, T0Packed.Residual r) {
            gemm(task + "_h", a, r.hidden(), scratch, m, Kernels.RELU);
            gemm(task + "_o", scratch, r.output(), o, m, 0);
            gemm(task + "_r", a, r.residual(), o, m, Kernels.ACCUMULATE);
        }

        /** {@code C = act(A·Wᵀ + b) (+ C)}, with the flags of {@link Kernels#gemm}, in the plan's GEMM mode. */
        void gemm(String task, FloatArray a, T0Packed.Lin l, FloatArray c, int m, int flags) {
            if (gemmMode.cublas()) {
                cublas(task, a, l, c, m, flags);
                written(c);
                return;
            }
            int f = (flags & ~NO_BIAS) | Kernels.BIAS;
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
                    g.task(task + "_sum", Kernels::splitKSum, ctx, pa, slices, w, l.b(), c, m, l.out(), f);
                    written(c);
                    return;
                }
                if (wide) {
                    g.task(task, Kernels::gemmSimdgroupWide, ctx, a, 0, w, l.w(), l.b(), c, m, l.out(), l.in(), f, l.in());
                } else {
                    g.task(task, Kernels::gemmSimdgroup, ctx, a, 0, w, l.w(), l.b(), c, m, l.out(), l.in(), f);
                }
            } else if (gemmMode == GemmMode.HALF) {
                HalfFloatArray ah = half(task, a, m * l.in());
                HalfFloatArray wh = weights.halfMatrix(l);
                weight(wh);
                int t = Kernels.HGEMM_TILE, threads = Kernels.HGEMM_THREADS;
                wg = new WorkerGrid2D((l.out() + t - 1) / t * threads, (m + t - 1) / t * threads);
                wg.setLocalWork(threads, threads, 1);
                g.task(task, Kernels::gemmHalf, ctx, ah, wh, c, w, l.b(), m, l.out(), l.in(), f, noTwin, 0);
            } else if (gemmMode == GemmMode.BLOCKED && m >= blockedWideMinRows) {
                int tr = Kernels.WIDE_ROWS, tc = Kernels.WIDE_COLS, threads = Kernels.HGEMM_THREADS;
                wg = new WorkerGrid2D((l.out() + tc - 1) / tc * threads, (m + tr - 1) / tr * threads);
                wg.setLocalWork(threads, threads, 1);
                g.task(task, Kernels::gemmBlockedWide, ctx, a, 0, w, l.w(), l.b(), c, 0, m, l.out(), l.in(), f);
            } else if (gemmMode == GemmMode.BLOCKED && m >= blockedMinRows) {
                int t = Kernels.HGEMM_TILE, threads = Kernels.HGEMM_THREADS;
                wg = new WorkerGrid2D((l.out() + t - 1) / t * threads, (m + t - 1) / t * threads);
                wg.setLocalWork(threads, threads, 1);
                g.task(task, Kernels::gemmBlocked, ctx, a, 0, w, l.w(), l.b(), c, 0, m, l.out(), l.in(), f);
            } else {
                int tile = Kernels.TILE;
                wg = new WorkerGrid2D((l.out() + tile - 1) / tile * tile, (m + tile - 1) / tile * tile);
                wg.setLocalWork(tile, tile, 1);
                g.task(task, Kernels::gemm, ctx, a, 0, w, l.w(), l.b(), c, 0, m, l.out(), l.in(), f);
            }
            grid.addWorkerGrid(name + "." + task, wg);
            written(c);
        }

        /**
         * The GEMM in cuBLAS: row-major {@code C[m, n] = A[m, k]·W[n, k]ᵀ} as the column-major
         * {@code gemm(T, N, n, m, k, W, k, A, k, C, n)}. The bias is written into C first (then β = 1), except
         * before a ReLU, where one epilogue adds the bias and applies the ReLU after the GEMM, and with
         * {@link #NO_BIAS}, where the next kernel adds it.
         */
        private void cublas(String task, FloatArray a, T0Packed.Lin l, FloatArray c, int m, int flags) {
            int nOut = l.out(), k = l.in();
            boolean acc = (flags & Kernels.ACCUMULATE) != 0;
            boolean relu = (flags & Kernels.RELU) != 0;
            if (relu && acc) {
                throw new IllegalStateException("GEMM " + task + ": ReLU after accumulation is not supported");
            }
            boolean bias = (flags & NO_BIAS) == 0;
            if (!relu && bias) {
                items(task + "_bias", m * nOut);
                g.task(task + "_bias", Kernels::biasFill, ctx, c, w, l.b(), m, nOut, acc ? 1 : 0);
            }
            float beta = relu ? 0f : bias || acc ? 1f : 0f;
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
            if (relu) {
                items(task + "_relu", m * nOut);
                g.task(task + "_relu", Kernels::biasRelu, ctx, c, w, l.b(), m, nOut, noTwin, 0);
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

        /** RMSNorm of {@code in} into {@code o}, after adding a pending bias to {@code in} when {@code bias >= 0}. */
        void norm(String task, FloatArray in, FloatArray o, int bias, int wOff, int rows) {
            if (bias < 0) {
                rms(task, in, o, wOff, rows);
                return;
            }
            grid.addWorkerGrid(name + "." + task, group(rows * T0Kernels.NORM_THREADS, T0Kernels.NORM_THREADS));
            g.task(task, T0Kernels::biasRmsNorm, ctx, in, o, w, bias, wOff, config.embedDim(), T0Config.NORM_EPS, rows);
            written(in);
            written(o);
        }

        void rms(String task, FloatArray in, FloatArray o, int wOff, int rows) {
            grid.addWorkerGrid(name + "." + task, group(rows * Kernels.NORM_THREADS, Kernels.NORM_THREADS));
            g.task(task, Kernels::rmsNorm, ctx, in, o, w, wOff, config.embedDim(), T0Config.NORM_EPS, noTwin, 0, rows);
            written(o);
        }

        void items(String task, int count) {
            grid.addWorkerGrid(name + "." + task, group((count + ITEM_GROUP - 1) / ITEM_GROUP * ITEM_GROUP, ITEM_GROUP));
        }

        @Override
        public long deviceBytes() {
            // TornadoVM counts the consumed weights in every plan; they are on the device once
            return Math.max(0, exec.getCurrentDeviceMemoryUsage() - SharedWeights.size(planWeights));
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
