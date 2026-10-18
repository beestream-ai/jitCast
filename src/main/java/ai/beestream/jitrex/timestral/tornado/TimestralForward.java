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
package ai.beestream.jitrex.timestral.tornado;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitrex.timestral.BlockForward;
import ai.beestream.jitrex.timestral.BlockInput;
import ai.beestream.jitrex.timestral.BlockLayout;
import ai.beestream.jitrex.timestral.BlockPrep;
import ai.beestream.jitrex.timestral.Quantiles;
import ai.beestream.jitrex.timestral.TimestralConfig;
import ai.beestream.jitrex.timestral.cpu.CpuTimestral;
import ai.beestream.jitrex.tornado.Kernels;

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

/**
 * {@code TimestralModel.predict_block} on a TornadoVM device: the scaler and the patch tokens, and
 * the quantile head's monotone fan-out and denormalization run on the host ({@link BlockPrep}); the
 * patch embedding, the transformer and the head's projections run as one task graph per block
 * shape (rows, encoder tokens, horizon patches, covariates), compiled once and cached.
 *
 * <p>The device is {@code -Dtirex.tornado.device=<driver>:<device>}, or the default one. GEMMs run
 * in FP32: on the shared register-blocked kernel from {@code -Dtimestral.tornado.blockedMinRows}
 * rows (128 by default; its 64-row tiles waste most of their work below that) and on the shared
 * tiled one under it, or on the tiled one only with {@code -Dtimestral.tornado.gemm=tiled}.
 */
public final class TimestralForward implements BlockForward, AutoCloseable {

    private static final String G = "ts";

    /** Work-group size of the element-wise kernels. */
    private static final int ITEM_GROUP = 128;

    /** Whether GEMMs may use {@link Kernels#gemmBlocked} rather than {@link Kernels#gemm}. */
    private final boolean blockedGemm = !"tiled".equalsIgnoreCase(System.getProperty("timestral.tornado.gemm", "blocked"));
    /** The fewest rows a GEMM needs to run on {@link Kernels#gemmBlocked}. */
    private final int blockedMinRows = Integer.getInteger("timestral.tornado.blockedMinRows", 128);

    /** One compiled block shape. */
    public record Shape(int rows, int encoder, int horizon, boolean covariates) {}

    private final TimestralConfig config;
    private final TimestralPacked weights;
    private final Quantiles quantiles;
    private final float[] invFreq;
    private final int maxCachedPlans;
    private final Map<Shape, Plan> plans = new LinkedHashMap<>(16, 0.75f, true);
    private TornadoDevice device;

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
    }

    private TornadoDevice device() {
        if (device == null) {
            String prop = System.getProperty("tirex.tornado.device");
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
        BlockLayout lay = BlockLayout.of(config, block);
        BlockPrep prep = BlockPrep.of(config, lay);
        Shape shape = new Shape(lay.rows(), lay.encoderTokens(), lay.horizonPatches(), lay.covariates());
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
        float[] raw = plan.run(lay, prep);
        return prep.finish(config, lay, quantiles, raw);
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
                        grid.addWorkerGrid(G + "." + s + "attn", group(rows * heads * qTiles * TimestralKernels.ATTN_THREADS,
                                TimestralKernels.ATTN_THREADS));
                        g.task(s + "attn", TimestralKernels::timeAttention, ctx, big, att, keyValid, known, rows, enc,
                                enc - (shape.covariates() ? h : 0), n, heads, dh, d);
                    } else {
                        grid.addWorkerGrid(G + "." + s + "attn", group(n * heads * rTiles * TimestralKernels.ATTN_THREADS,
                                TimestralKernels.ATTN_THREADS));
                        g.task(s + "attn", TimestralKernels::variateAttention, ctx, big, att, keyValid, groupStart, groupEnd,
                                role, shape.covariates() ? 1 : 0, rows, enc, n, heads, dh, d);
                    }
                    if (a.gate() != null) {
                        gemm(s + "gproj", hn, a.gate(), gate, tokens, 0);
                        items(s + "gate", tokens * d);
                        g.task(s + "gate", TimestralKernels::gate, ctx, att, gate, tokens * d);
                    }
                    gemm(s + "wo", att, a.wo(), x, tokens, Kernels.ACCUMULATE);
                }
                String s = "l" + li + "_";
                rms(s + "mnorm", x, hn, layer.mlpNorm(), tokens, c.normEps());
                gemm(s + "up", hn, layer.up(), big, tokens, 0);
                items(s + "swiglu", tokens * f);
                g.task(s + "swiglu", Kernels::swiGlu, ctx, big, act, tokens, f, noTwin, 0);
                gemm(s + "down", act, layer.down(), x, tokens, Kernels.ACCUMULATE);
            }
            rms("onorm", x, y, weights.outNorm(), tokens, c.normEps());
            items("gather", decTokens * d);
            g.task("gather", TimestralKernels::gatherDecoder, ctx, y, dec, rows, enc, n, h, d);
            residual("head", dec, big, raw, decTokens, weights.head());
            g.transferToHost(DataTransferMode.EVERY_EXECUTION, raw);

            exec = new TornadoExecutionPlan(g.snapshot());
            exec.withGridScheduler(grid);
            exec.withDevice(device());
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

        void gemm(String name, FloatArray a, TimestralPacked.Lin l, FloatArray c, int m, int flags) {
            int f = flags | (l.b() >= 0 ? Kernels.BIAS : 0);
            WorkerGrid wg;
            if (blockedGemm && m >= blockedMinRows) {
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
        }

        void rms(String name, FloatArray in, FloatArray out, int wOff, int rows, float eps) {
            grid.addWorkerGrid(G + "." + name, group(rows * Kernels.NORM_THREADS, Kernels.NORM_THREADS));
            g.task(name, Kernels::rmsNorm, ctx, in, out, w, wOff, config.dModel(), eps, noTwin, 0);
        }

        void items(String task, int count) {
            grid.addWorkerGrid(G + "." + task, group((count + ITEM_GROUP - 1) / ITEM_GROUP * ITEM_GROUP, ITEM_GROUP));
        }

        /** Uploads one block's per-call inputs, runs the graph, returns the head's raw outputs. */
        float[] run(BlockLayout lay, BlockPrep prep) {
            int rows = shape.rows(), enc = shape.encoder(), h = shape.horizon(), n = enc + h, dh = config.headDim();
            float[] t = prep.tokens();
            for (int i = 0; i < t.length; i++) {
                tok.set(i, t[i]);
            }
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
            exec.execute();
            float[] out = new float[raw.getSize()];
            for (int i = 0; i < out.length; i++) {
                out[i] = raw.get(i);
            }
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
