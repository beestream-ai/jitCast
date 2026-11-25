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
package ai.beestream.jitcast.toto.tornado;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitcast.toto.TotoConfig;
import ai.beestream.jitcast.toto.TotoForward;
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
 * Toto 2.0's forward pass on a TornadoVM device: one compiled plan per {@link Shape}, kept in a
 * {@link PlanCache}, all plans consuming one device copy of the weights ({@link SharedWeights}). The GEMMs
 * follow {@link GemmMode} ({@code -Dtoto.tornado.gemm}); attention, the rotary embedding and the residual
 * mixing are {@link TotoKernels}.
 */
public final class TornadoTotoForward implements TotoForward {

    static {
        Devices.defaults();
    }

    private static final String G = "toto";
    private static final int ITEM_GROUP = 128;
    private static final double XPOS_SCALE_BASE = 256;
    private static final double ROPE_BASE = 10000;

    private final int blockedMinRows = Options.integer("toto.tornado.blockedMinRows", 128, 1);
    private final int blockedWideMinRows = Options.integer("toto.tornado.wideMinRows", 1024, 1);
    private final int simdgroupWideMinRows = Options.integer("toto.tornado.simdgroupWideMinRows", 64, 1);

    /**
     * A compiled plan's batch shape.
     *
     * @param rows the variate rows
     * @param patches the patches per row
     * @param forecast the forecast patches
     */
    public record Shape(int rows, int patches, int forecast) {}

    private final TotoConfig config;
    private final TotoPacked weights;
    private final PlanCache<Shape, Plan> plans;
    private final TornadoDevice device = Devices.device();
    private final SharedWeights shared = new SharedWeights(G, device);
    private final GemmMode gemmMode;
    private final boolean cudaGraph;

    /**
     * A forward pass caching up to {@code toto.tornado.plans} (8) compiled shapes.
     *
     * @param config the checkpoint's configuration
     * @param weights its packed weights
     */
    public TornadoTotoForward(TotoConfig config, TotoPacked weights) {
        this(config, weights, Options.integer("toto.tornado.plans", 8, 1));
    }

    /**
     * A forward pass.
     *
     * @param config the checkpoint's configuration
     * @param weights its packed weights
     * @param maxCachedPlans the compiled shapes to keep
     */
    public TornadoTotoForward(TotoConfig config, TotoPacked weights, int maxCachedPlans) {
        if (config.headDim() > TotoKernels.HEAD_DIM) {
            throw new IllegalArgumentException("attention heads wider than " + TotoKernels.HEAD_DIM + " are not supported");
        }
        this.config = config;
        this.weights = weights;
        this.gemmMode = Options.enumeration("toto.tornado.gemm", GemmMode.AUTO, GemmMode.class).resolve(device);
        long weightBytes = SharedWeights.estimate(weights.data().getNumBytesOfSegment(),
                gemmMode == GemmMode.CUBLAS || gemmMode == GemmMode.CUBLAS_TF32, gemmMode == GemmMode.CUBLAS_FP16 || gemmMode == GemmMode.HALF);
        this.plans = new PlanCache<>(Math.max(1, maxCachedPlans),
                Options.bytes("toto.tornado.planBytes", Math.max(1, Devices.memory() - weightBytes), 1));
        this.cudaGraph = Options.bool("toto.tornado.cudaGraph", gemmMode.cublas());
    }

    /** {@return the GEMM mode in use} */
    public GemmMode gemmMode() {
        return gemmMode;
    }

    /** {@return the shapes compiled and cached, least recently used first} */
    public List<Shape> cachedShapes() {
        return plans.keys();
    }

    @Override
    public float[] forward(Input in) {
        Shape shape = new Shape(in.rows(), in.patches(), in.forecast());
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

    /** {@return the rotation table of a shape}: per patch and pair, cos, sin and the query's and key's xPos scales */
    float[] rotation(int n) {
        int w = config.ropeWidth(), pairs = w / 2, center = n / 2;
        float[] rot = new float[n * pairs * TotoKernels.ROT];
        for (int t = 0; t < n; t++) {
            double rel = (t - center) / XPOS_SCALE_BASE;
            for (int k = 0; k < pairs; k++) {
                double angle = t * (1.0 / Math.pow(ROPE_BASE, (2.0 * k) / w));
                double base = (2.0 * k + 0.4 * w) / (1.4 * w);
                int o = (t * pairs + k) * TotoKernels.ROT;
                rot[o] = (float) Math.cos(angle);
                rot[o + 1] = (float) Math.sin(angle);
                rot[o + 2] = (float) Math.pow(base, rel);
                rot[o + 3] = (float) Math.pow(base, -rel);
            }
        }
        return rot;
    }

    private final class Plan implements PlanCache.Plan {
        final Shape shape;
        final FloatArray w = weights.data();
        final GridScheduler grid = new GridScheduler();
        final KernelContext ctx = new KernelContext();
        final Set<Object> planWeights = Collections.newSetFromMap(new IdentityHashMap<>());
        final Set<Object> declared = Collections.newSetFromMap(new IdentityHashMap<>());
        final Map<FloatArray, HalfFloatArray> halves = new IdentityHashMap<>();
        final Set<FloatArray> fresh = Collections.newSetFromMap(new IdentityHashMap<>());
        final HalfFloatArray noTwin = new HalfFloatArray(1);
        final TaskGraph g = new TaskGraph(G);
        final FloatArray tok, x, hn, big, att, y, gl, dec, hb, out, rot;
        final IntArray valid, first, count;
        final TornadoExecutionPlan exec;

        Plan(Shape shape) {
            this.shape = shape;
            TotoConfig c = config;
            int rows = shape.rows(), n = shape.patches(), k = shape.forecast();
            int d = c.dModel(), ff = c.dFf(), p = c.patchSize(), heads = c.numHeads(), hd = c.headDim(), pairs = c.ropeWidth() / 2;
            int tokens = rows * n, heads2 = rows * k;
            tok = new FloatArray(tokens * 2 * p);
            x = new FloatArray(tokens * d);
            hn = new FloatArray(tokens * d);
            big = new FloatArray(tokens * Math.max(4 * d, Math.max(3 * d, 2 * ff)));
            att = new FloatArray(tokens * d);
            y = new FloatArray(tokens * d);
            gl = new FloatArray(tokens * ff);
            dec = new FloatArray(heads2 * d);
            hb = new FloatArray(heads2 * 4 * d);
            out = new FloatArray(heads2 * p * c.quantiles());
            rot = FloatArray.fromArray(rotation(n));
            valid = new IntArray(tokens);
            first = new IntArray(rows);
            count = new IntArray(rows);
            weight(w);
            g.transferToDevice(DataTransferMode.FIRST_EXECUTION, x, hn, big, att, y, gl, dec, hb, out, rot, noTwin);
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, tok, valid, first, count);
            residual("pp", tok, big, x, tokens, weights.patchProj());
            for (int li = 0; li < weights.layers().size(); li++) {
                TotoPacked.Layer l = weights.layers().get(li);
                String s = "l" + li + "_";
                rms(s + "n1", x, hn, tokens);
                gemm(s + "qkv", hn, l.inProj(), big, tokens, 0);
                if (c.variateLayer(li)) {
                    grid.addWorkerGrid(G + "." + s + "attn", group(tokens * heads * TotoKernels.VAR_THREADS, TotoKernels.VAR_THREADS));
                    g.task(s + "attn", TotoKernels::variateAttention, ctx, big, att, valid, first, count, rows, n, heads, hd, d);
                } else {
                    items(s + "rope", tokens * heads * pairs * 2);
                    g.task(s + "rope", TotoKernels::rope, ctx, big, rot, rows, n, heads, hd, d, pairs);
                    int tiles = (n + TotoKernels.ATTN_QUERIES - 1) / TotoKernels.ATTN_QUERIES;
                    grid.addWorkerGrid(G + "." + s + "attn", group(rows * heads * tiles * TotoKernels.ATTN_THREADS, TotoKernels.ATTN_THREADS));
                    g.task(s + "attn", TotoKernels::timeAttention, ctx, big, att, valid, rows, n, heads, hd, d);
                }
                written(big);
                written(att);
                gemm(s + "o", att, l.outProj(), y, tokens, 0);
                mix(s + "ra", y, l.attnKeep(), tokens * d);
                rms(s + "n2", x, hn, tokens);
                gemm(s + "fc1", hn, l.fc1(), big, tokens, 0);
                items(s + "glu", tokens * ff);
                g.task(s + "glu", Kernels::swiGlu, ctx, big, gl, tokens, ff, noTwin, 0);
                written(gl);
                gemm(s + "fc2", gl, l.fc2(), y, tokens, 0);
                mix(s + "rf", y, l.mlpKeep(), tokens * d);
            }
            rms("on", x, hn, tokens);
            items("gather", heads2 * d);
            g.task("gather", TotoKernels::gather, ctx, hn, dec, rows, n, k, d);
            written(dec);
            residual("hd", dec, hb, out, heads2, weights.head());
            g.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            shared.consume(g, new ArrayList<>(declared));
            exec = new TornadoExecutionPlan(g.snapshot());
            exec.withGridScheduler(grid);
            exec.withDevice(device);
            if (cudaGraph) {
                exec.withCUDAGraph();
            }
            if (Options.bool("toto.tornado.profile", false)) {
                exec.withProfiler(uk.ac.manchester.tornado.api.enums.ProfilerMode.CONSOLE);
            }
        }

        float[] run(Input in) {
            int p = config.patchSize(), tokens = in.rows() * in.patches();
            float[] t = new float[tokens * 2 * p];
            for (int i = 0; i < tokens; i++) {
                System.arraycopy(in.values(), i * p, t, i * 2 * p, p);
                System.arraycopy(in.missing(), i * p, t, i * 2 * p + p, p);
            }
            MemorySegment.copy(t, 0, tok.getSegment(), ValueLayout.JAVA_FLOAT, 0, t.length);
            for (int i = 0; i < tokens; i++) {
                valid.set(i, in.valid()[i] ? 1 : 0);
            }
            for (int s = 0; s < in.series(); s++) {
                int a = in.seriesStart()[s], b = in.seriesStart()[s + 1];
                for (int r = a; r < b; r++) {
                    first.set(r, a);
                    count.set(r, b - a);
                }
            }
            exec.execute();
            float[] res = new float[out.getSize()];
            MemorySegment.copy(out.getSegment(), ValueLayout.JAVA_FLOAT, 0, res, 0, res.length);
            return res;
        }

        private void weight(Object buf) {
            if (declared.add(buf)) {
                planWeights.add(buf);
            }
        }

        void residual(String task, FloatArray a, FloatArray scratch, FloatArray c, int m, TotoPacked.Mlp r) {
            gemm(task + "_h", a, r.linear1(), scratch, m, Kernels.SILU);
            gemm(task + "_o", scratch, r.linear2(), c, m, 0);
            gemm(task + "_s", a, r.skip(), c, m, Kernels.ACCUMULATE);
        }

        void mix(String task, FloatArray branch, float keep, int n) {
            items(task, n);
            g.task(task, TotoKernels::mix, ctx, x, branch, keep, n);
            written(x);
        }

        void rms(String task, FloatArray in, FloatArray o, int rows) {
            grid.addWorkerGrid(G + "." + task, group(rows * Kernels.NORM_THREADS, Kernels.NORM_THREADS));
            g.task(task, Kernels::rmsNorm, ctx, in, o, w, weights.ones(), config.dModel(), config.normEps(), noTwin, 0, rows);
            written(o);
        }

        void items(String task, int n) {
            grid.addWorkerGrid(G + "." + task, group((n + ITEM_GROUP - 1) / ITEM_GROUP * ITEM_GROUP, ITEM_GROUP));
        }

        void gemm(String task, FloatArray a, TotoPacked.Lin l, FloatArray c, int m, int flags) {
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
            grid.addWorkerGrid(G + "." + task, wg);
            written(c);
        }

        private void cublas(String task, FloatArray a, TotoPacked.Lin l, FloatArray c, int m, int flags) {
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

        private HalfFloatArray half(String task, FloatArray a, int n) {
            HalfFloatArray h = halves.get(a);
            if (h == null) {
                h = new HalfFloatArray(a.getSize());
                halves.put(a, h);
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, h);
            }
            if (fresh.add(a)) {
                items(task + "_h", n);
                g.task(task + "_h", Kernels::toHalf, ctx, a, h, n);
            }
            return h;
        }

        private void written(FloatArray buf) {
            fresh.remove(buf);
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
