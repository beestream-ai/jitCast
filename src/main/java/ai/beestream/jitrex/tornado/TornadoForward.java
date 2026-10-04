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
package ai.beestream.jitrex.tornado;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitrex.model.ForwardPass;
import ai.beestream.jitrex.model.ModelBatch;
import ai.beestream.jitrex.model.TiRex2Config;

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

    static {
        tornadoDefaults();
    }

    /**
     * Sets jitrex's TornadoVM defaults, unless already set. TornadoVM reads them when its backends
     * start, so code that calls TornadoVM before creating a {@code TornadoForward} calls this first.
     * TornadoVM's Metal backend spin-waits up to 1 ms on each dispatch before it blocks; a forward
     * pass runs dozens of kernels per task graph, so its waits outlast the spin and jitrex runs up
     * to 2× slower on Metal. The spin is off.
     */
    public static void tornadoDefaults() {
        if (System.getProperty("tornado.metal.dispatch.spinWait") == null) {
            System.setProperty("tornado.metal.dispatch.spinWait", "False");
        }
    }

    private static final String G = "tirex";

    /** Work-group size of the element-wise kernels. */
    private static final int ITEM_GROUP = 128;

    /**
     * How GEMMs run: the portable tiled kernel, or on CUDA cuBLAS in FP32, TF32, or FP16 tensor
     * cores ({@code cublasGemmExFP16FP32}: FP16 weights and activations, FP32 accumulation and
     * output, so the residual stream stays FP32). {@code CUBLAS_AUTO} picks per GEMM: FP16 once the
     * row count reaches {@code -Dtirex.tornado.fp16MinTokens}, FP32 below it. The default, 0, is
     * FP16 everywhere: with CUDA 13 cuBLAS the FP16 tensor-core GEMM is faster at every row count
     * (−30% at 1 series, −44% at 8 on an RTX 5070 Ti). The threshold dates from CUDA 12.0 cuBLAS,
     * whose FP16→FP32 GEMM fell back to a non-tensor-core kernel at small row counts.
     * {@code HALF} and {@code CUTILE} are the same FP16 GEMM written in TornadoVM, for every row
     * count: {@code HALF} with KernelContext ({@link TiRexKernels#gemmHalf}, any backend) and
     * {@code CUTILE} with the CUDA Tile API ({@link TiRexKernels#gemmHalfTile}, tensor cores).
     */
    public enum GemmMode {
        TILED, CUBLAS, CUBLAS_TF32, CUBLAS_FP16, CUBLAS_AUTO, HALF, CUTILE;

        boolean cublas() {
            return this == CUBLAS || this == CUBLAS_TF32 || this == CUBLAS_FP16 || this == CUBLAS_AUTO;
        }

        static GemmMode fromProperty() {
            String v = System.getProperty("tirex.tornado.gemm", "tiled").trim().toUpperCase().replace('-', '_');
            return GemmMode.valueOf(v);
        }
    }

    private final GemmMode gemmMode = GemmMode.fromProperty();

    /** {@code -Dtirex.tornado.fuse=false} turns the fused SiLU epilogue and residual+RMSNorm kernels off (A/B). */
    private final boolean fuse = !"false".equalsIgnoreCase(System.getProperty("tirex.tornado.fuse", "true"));

    /**
     * Keep the FFN's intermediates in FP16 only ({@code -Dtirex.tornado.ffnHalf}, on by default):
     * the up-projection writes FP16 (FP32 accumulation), SwiGLU reads and writes FP16, and the
     * down-projection reads that directly. The wide FP32 buffers then no longer hold the FFN, so
     * {@code proj} and {@code big} become one buffer. Applies to the FP16 cuBLAS modes and
     * {@code half}; CUDA Tile and the FP32 modes keep the FP32 FFN.
     */
    private final boolean ffnHalfEnabled = !"false".equalsIgnoreCase(System.getProperty("tirex.tornado.ffnHalf", "true"));

    /**
     * Capture and replay the plan's task graph as a CUDA graph ({@code -Dtirex.tornado.cudaGraph}). On
     * by default with the cuBLAS (CUDA-only) GEMM modes: −11% at batch 1 on an RTX 5070 Ti, with
     * identical results; off with the portable tiled GEMMs, which also run on OpenCL and Metal.
     */
    private final boolean cudaGraph = Boolean.parseBoolean(
            System.getProperty("tirex.tornado.cudaGraph", String.valueOf(gemmMode.cublas() || gemmMode == GemmMode.CUTILE)));

    private final int fp16MinTokens = Integer.getInteger("tirex.tornado.fp16MinTokens", 0);

    /**
     * Which cuBLAS call runs an FP16 GEMM: {@code gemmex-mixed} ({@code cublasGemmEx}, FP16 in,
     * FP32 out — cuBLAS picks a non-tensor-core kernel for it on sm_120), {@code gemmex-h16}
     * ({@code cublasGemmEx} all FP16) or {@code lt-h16} (cuBLASLt all FP16); the last two write an
     * FP16 C that one epilogue kernel brings back to FP32 ({@code -Dtirex.tornado.fp16Kernel}).
     */
    enum Fp16Kernel {
        GEMMEX_MIXED, GEMMEX_H16, LT_H16
    }

    private final Fp16Kernel fp16Kernel = Fp16Kernel.valueOf(
            System.getProperty("tirex.tornado.fp16Kernel", "gemmex-mixed").trim().toUpperCase().replace('-', '_'));

    /** Whether a GEMM over {@code m} rows runs in FP16 under the current mode. */
    private boolean halfGemm(int m) {
        return gemmMode == GemmMode.CUBLAS_FP16 || gemmMode == GemmMode.HALF || gemmMode == GemmMode.CUTILE
                || (gemmMode == GemmMode.CUBLAS_AUTO && pad(m) >= fp16MinTokens);
    }

    /**
     * {@code -Dtirex.tornado.mlstm=auto|chunk|step}: the chunkwise mLSTM kernel (blocks of query
     * steps share each staged key; {@link TiRexKernels#mlstmChunk}) or the one-step-per-work-group
     * one ({@link TiRexKernels#mlstmCell}). {@code auto} (default) takes the chunkwise kernel unless
     * its grid would be smaller than {@link #MLSTM_CHUNK_MIN_GROUPS} work-groups: on an RTX 5070 Ti
     * it is 26–35% faster at 285 patches from 8 series up and 4–5% faster at 74 patches from 64
     * series, but 5% slower for a single 74-patch series, whose 20 work-groups leave the GPU idle.
     */
    private final String mlstmMode = System.getProperty("tirex.tornado.mlstm", "auto").trim().toLowerCase();

    static final int MLSTM_CHUNK_MIN_GROUPS = 64;

    private boolean mlstmChunk(int rows, int heads, int seq) {
        int blocks = (seq + TiRexKernels.MLSTM_TB - 1) / TiRexKernels.MLSTM_TB;
        return switch (mlstmMode) {
            case "chunk" -> true;
            case "step" -> false;
            default -> rows * heads * blocks >= MLSTM_CHUNK_MIN_GROUPS;
        };
    }

    /**
     * {@code -Dtirex.tornado.slstmR=auto|fp16-reg|fp16|fp32}: the precision of the sLSTM recurrent
     * weights, and where they live. {@code fp16-reg} is FP16 with half of each thread's R values
     * held in registers where it pays (see {@link #registerRecurrent}); {@code fp16} streams FP16 R
     * every step; {@code fp32} streams FP32 R, for results closest to FP32. {@code auto} (default)
     * is {@code fp16-reg} on the devices where that kernel is known to fit
     * ({@link #registerKernelFits}), else {@code fp16}.
     */
    private final String slstmRProperty = System.getProperty("tirex.tornado.slstmR", "auto").trim().toLowerCase();
    private final boolean halfRecurrent = !"fp32".equals(slstmRProperty);
    private String slstmR;

    private String slstmR() {
        if (slstmR == null) {
            slstmR = "auto".equals(slstmRProperty) ? (registerKernelFits(device()) ? "fp16-reg" : "fp16") : slstmRProperty;
        }
        return slstmR;
    }

    /**
     * Whether the register-resident sLSTM kernel fits the device: it needs 512 threads per work-group
     * with up to 128 registers each, and whether the compiler gets it under that depends on the
     * target. It does on CUDA compute capability 12 (RTX 5070 Ti) and on Apple GPUs (Metal and
     * OpenCL); on an A100 (8.0) CUDA refuses to launch it and NVIDIA's OpenCL returns wrong results
     * without an error. OpenCL does not report the compute capability, so only Apple's is trusted.
     */
    static boolean registerKernelFits(uk.ac.manchester.tornado.api.common.TornadoDevice d) {
        return switch (d.getTornadoVMBackend()) {
            case METAL -> true;
            case OPENCL -> d.getPlatformName() != null && d.getPlatformName().toLowerCase().contains("apple");
            case CUDA -> cudaComputeCapabilityMajor(d) >= 12;
            default -> false;
        };
    }

    /** The CUDA device's compute capability major version, from its device version ("CUDA 8.0"); 0 if unknown. */
    static int cudaComputeCapabilityMajor(uk.ac.manchester.tornado.api.common.TornadoDevice d) {
        String info = String.valueOf(d.getPhysicalDevice().getDeviceInfo());
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("Device version\\s*:\\s*CUDA (\\d+)\\.").matcher(info);
        return m.find() ? Integer.parseInt(m.group(1)) : 0;
    }

    /**
     * Whether the configuration computes anything below FP32 (FP16 or TF32 GEMM inputs, FP16 sLSTM
     * weights): results then differ from the FP32 reference by up to ~2e-3 instead of ~1e-5.
     */
    public static boolean reducedPrecision() {
        String gemm = System.getProperty("tirex.tornado.gemm", "tiled").trim().toLowerCase();
        String r = System.getProperty("tirex.tornado.slstmR", "auto").trim().toLowerCase();
        return !(gemm.equals("tiled") || gemm.equals("cublas")) || !r.equals("fp32");
    }

    /**
     * {@code -Dtirex.tornado.slstmR=fp16-reg}: FP16 R with half of each thread's R values held in
     * registers for the whole sequence ({@link TiRexKernels#slstmRecurrenceRegR}); it needs
     * 512-thread work-groups, and falls back to {@code fp16} on devices that cap them lower.
     */
    private boolean registerRecurrent(int dh, int streamedRowsPerGroup) {
        return "fp16-reg".equals(slstmR()) && dh == 128 && slstmGatesPerThread(dh) == 1
                && streamedRowsPerGroup < TiRexKernels.SLSTM_MAX_ROWS;
    }

    /**
     * Rows per work-group the register-resident sLSTM kernel can take: with 4 or more, its 512 threads ask
     * for more registers than an SM has (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES on an RTX 5070 Ti, and
     * CL_OUT_OF_RESOURCES on NVIDIA's OpenCL). So it only replaces the streamed kernel where that
     * one would take fewer than {@link TiRexKernels#SLSTM_MAX_ROWS} rows per group: on an RTX 5070 Ti
     * it is 14–23% faster end to end up to about 64 rows, and 3% slower where the streamed kernel
     * shares each R read between 8 rows (the largest batches).
     */
    static final int SLSTM_REG_MAX_ROWS = 2;

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

    /** Work-groups the sLSTM recurrence aims for (tuned on an RTX 5070 Ti: 4 rows per group at batch 64, 8 at 512). */
    private static final int SLSTM_TARGET_GROUPS = 64;

    /**
     * Rows per sLSTM work-group: the largest power of two up to {@link TiRexKernels#SLSTM_MAX_ROWS}
     * that still yields {@link #SLSTM_TARGET_GROUPS} work-groups, so small batches keep their
     * parallelism and large ones share each R read across rows. {@code -Dtirex.tornado.slstmRows}
     * overrides it.
     */
    static int slstmRowsPerGroup(int rows, int heads) {
        int forced = Integer.getInteger("tirex.tornado.slstmRows", 0);
        if (forced > 0) {
            return Math.min(forced, TiRexKernels.SLSTM_MAX_ROWS);
        }
        int rb = 1;
        while (rb * 2 <= TiRexKernels.SLSTM_MAX_ROWS && (rows + 2 * rb - 1) / (2 * rb) * heads >= SLSTM_TARGET_GROUPS) {
            rb *= 2;
        }
        return rb;
    }

    /**
     * {@code -Dtirex.tornado.rowBuckets=off|pow2}: with {@code pow2}, round each batch's row count up
     * to a power of two, so batch sizes share a handful of plans instead of building one per size
     * (a new plan costs 0.2–0.6 s with cuBLAS, 0.5–1.3 s with the portable GEMMs). The extra rows are
     * copies of row 0, each alone in its group, and their outputs are dropped. Off by default: the
     * padding rows are real work, 20–60% on every call of a size that is not a power of two (RTX
     * 5070 Ti, cublas-auto), so it only pays for workloads with many sizes each seen a few times.
     */
    private final boolean rowBuckets = "pow2".equalsIgnoreCase(System.getProperty("tirex.tornado.rowBuckets", "off"));

    /** The row count a batch of {@code rows} is padded to. */
    static int rowBucket(int rows) {
        return rows <= 1 ? 1 : Integer.highestOneBit(rows - 1) << 1;
    }

    /**
     * The batch with {@code padded − rows} extra rows appended: copies of row 0, targets, each in a
     * group of its own, so the known covariates and the multi-row groups (the rest of the plan's
     * shape) are unchanged.
     */
    static ModelBatch padRows(ModelBatch batch, int padded) {
        int rows = batch.rows();
        float[][] x = java.util.Arrays.copyOf(batch.x(), padded);
        float[] group = java.util.Arrays.copyOf(batch.group(), padded);
        boolean[] target = java.util.Arrays.copyOf(batch.target(), padded);
        float next = 0f;
        for (float g : batch.group()) {
            next = Math.max(next, g + 1f);
        }
        for (int r = rows; r < padded; r++) {
            x[r] = batch.x()[0];
            group[r] = next + (r - rows);
            target[r] = true;
        }
        return new ModelBatch(x, group, target);
    }

    @Override
    public float[][][] forward(ModelBatch batch) {
        int rows = batch.rows();
        if (rowBuckets && rowBucket(rows) != rows) {
            return java.util.Arrays.copyOf(forwardExact(padRows(batch, rowBucket(rows))), rows);
        }
        return forwardExact(batch);
    }

    private float[][][] forwardExact(ModelBatch batch) {
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

    /** Row alignment of GEMM operands: cuBLAS uses FP16 tensor cores only for multiples of 8. */
    static final int GEMM_ROW_ALIGN = 8;

    static int pad(int rows) {
        return (rows + GEMM_ROW_ALIGN - 1) / GEMM_ROW_ALIGN * GEMM_ROW_ALIGN;
    }

    /** The widest GEMM output (N) of the model, which sizes the FP16 result scratch. */
    private int maxGemmOutput() {
        int n = Math.max(weights.inputEmbedding().hidden().out(), weights.outputEmbedding().hidden().out());
        for (PackedWeights.Block blk : weights.blocks()) {
            n = Math.max(n, blk.upGateUp().out());
            n = Math.max(n, blk.mlstm() ? blk.mlstmProj().out() : blk.slstmGates().out());
            n = Math.max(n, Math.max(blk.variate().qkv().out(), blk.variate().ffnIn().out()));
        }
        return n;
    }

    /** Whether the fused per-head-norm + residual + RMSNorm kernel fits: one thread per element of a row. */
    private boolean fusedNorm(int heads, int dh) {
        return heads * dh == TiRexKernels.FUSED_NORM_THREADS && device().getDeviceMaxWorkgroupDimensions()[0] >= heads * dh;
    }

    /** qkNorm vectors reduced concurrently: as many 128-thread segments as the device allows, up to 2·heads. */
    private int qkNormVectorsPerPass(int heads, int dh) {
        long max = Math.min(device().getDeviceMaxWorkgroupDimensions()[0], TiRexKernels.QKNORM_MAX_THREADS);
        int v = 1;
        while (v * 2 <= 2 * heads && (long) v * 2 * dh <= max) {
            v *= 2;
        }
        return v;
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
        final uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray cHalf, ffnUp, ffnSw;
        /** Stands in for C when a GEMM writes only its FP16 twin; never read or written. */
        final FloatArray noOut = new FloatArray(1);
        /** FP16 twins of the GEMM-input buffers, written by their producers (FP16 GEMM plans only). */
        final java.util.Map<FloatArray, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray> twins =
                new java.util.IdentityHashMap<>();
        /** Passed as the twin of a producer that has none; never written (flag 0). */
        final uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray noTwin = new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(1);
        final IntArray known, kept, groupStart, groupEnd, keptTarget;
        final TornadoExecutionPlan exec;
        final GridScheduler grid;

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
            boolean half = halfGemm(tokens);
            boolean ffnHalf = half && ffnHalfEnabled && (gemmMode.cublas() || gemmMode == GemmMode.HALF)
                    && weights.blocks().stream().allMatch(b -> b.upGateUp().b() < 0);
            int projWidth = Math.max(Math.max(mlstmProj, ffnHalf ? 0 : ffn), 3 * d);
            int bigWidth = Math.max(Math.max(ffnHalf ? 0 : 2 * ffn, 4 * d),
                    Math.max(Math.max(c.inputFfDim(), c.outputFfDim()), TiRex2Config.ATTENTION_FFN_FACTOR * d));

            x = new FloatArray(rows * len);
            present = new FloatArray(rows * len);
            // GEMM operands have their row count padded to a multiple of GEMM_ROW_ALIGN (see pad())
            int tokensP = pad(tokens);
            tok = new FloatArray(tokensP * 2 * p);
            stats = new FloatArray(rows * 3);
            h = new FloatArray(tokensP * d);
            n = new FloatArray(tokensP * d);
            cell = new FloatArray(tokensP * d);
            if (ffnHalf) {
                // without the FFN, no stage uses proj and big at the same time: one buffer serves both
                big = new FloatArray(tokensP * Math.max(projWidth, bigWidth));
                proj = big;
            } else {
                proj = new FloatArray(tokensP * projWidth);
                big = new FloatArray(tokensP * bigWidth);
            }
            int maxFfn = weights.blocks().stream().mapToInt(PackedWeights.Block::ffnDim).max().orElse(ffn);
            ffnUp = ffnHalf ? new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(tokensP * 2 * maxFfn) : null;
            ffnSw = ffnHalf ? new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(tokensP * maxFfn) : null;
            y = new FloatArray(tokensP * nq * p);
            out = new FloatArray(rows * nq * len);
            yc = nKnown > 0 ? new FloatArray(tokensP * d) : null;
            revIn = nKnown > 0 ? new FloatArray(pad(nKnown * seq) * d) : null;
            revOut = nKnown > 0 ? new FloatArray(pad(nKnown * seq) * d) : null;
            xk = nKept > 0 ? new FloatArray(pad(nKept * seq) * d) : null;
            cHalf = half && gemmMode.cublas() && fp16Kernel != Fp16Kernel.GEMMEX_MIXED
                    ? new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(tokensP * maxGemmOutput()) : null;
            known = nKnown > 0 ? new IntArray(nKnown) : null;
            kept = nKept > 0 ? new IntArray(nKept) : null;
            groupStart = nKept > 0 ? new IntArray(nKept) : null;
            groupEnd = nKept > 0 ? new IntArray(nKept) : null;
            keptTarget = nKept > 0 ? new IntArray(nKept) : null;
            FloatArray w = weights.data();

            KernelContext ctx = new KernelContext();
            GridScheduler grid = new GridScheduler();
            List<Object> sharedList = new ArrayList<>();
            java.util.Set<Object> listed = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
            for (Object o : new Object[] {w, x, present, tok, stats, h, n, cell, proj, big, y, out, noOut}) {
                if (listed.add(o)) {
                    sharedList.add(o);
                }
            }
            if (half) {
                // every buffer a GEMM reads as A gets an FP16 twin its producer fills, so no GEMM
                // input needs a conversion pass (see Builder.halfInput)
                for (FloatArray f : new FloatArray[] {tok, n, proj, big, cell, revIn}) {
                    if (f != null && !twins.containsKey(f)) {
                        twins.put(f, new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(f.getSize()));
                    }
                }
            }
            sharedList.add(noTwin);
            sharedList.addAll(twins.values());
            for (Object o : new Object[] {yc, revIn, revOut, xk, cHalf, ffnUp, ffnSw, known, kept, groupStart, groupEnd, keptTarget}) {
                if (o != null) {
                    sharedList.add(o);
                }
            }
            Object[] shared = sharedList.toArray();
            if (Boolean.getBoolean("tirex.tornado.memReport")) {
                memoryReport(tokens, new Object[] {"x", x, "present", present, "tok", tok, "stats", stats, "h", h, "n", n, "cell", cell,
                        "proj", proj, "big", big, "y", y, "out", out, "yc", yc, "revIn", revIn, "revOut", revOut, "xk", xk, "cHalf", cHalf,
                        "ffnUp", ffnUp, "ffnSw", ffnSw});
            }
            List<Object> perCall = new ArrayList<>(List.of(x, present));
            for (Object o : new Object[] {known, kept, groupStart, groupEnd, keptTarget}) {
                if (o != null) {
                    perCall.add(o);
                }
            }
            Builder b = new Builder(grid, ctx, w);
            b.twins = twins;
            b.noTwin = noTwin;
            b.cHalf = cHalf;
            b.noOut = noOut;

            // The whole forward pass is one task graph (input stage, the blocks, output stage). It has
            // more than 256 tasks and 8 KiB of bytecode, which needs TornadoVM 7.1.0 or newer.
            TaskGraph g = b.begin("in");
            g.transferToDevice(DataTransferMode.FIRST_EXECUTION, shared);
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, perCall.toArray());
            grid.addWorkerGrid(b.name + ".scale", group(rows * TiRexKernels.NORM_THREADS, TiRexKernels.NORM_THREADS));
            g.task("scale", TiRexKernels::scalePatch, ctx, x, present, tok, stats, rows, len, p, c.useArcsinh() ? 1 : 0,
                    c.binaryAware() ? 1 : 0, TiRex2Config.SCALER_EPS, b.twin(tok), b.hasTwin(tok));
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
                    g.task(s + "revIn", TiRexKernels::gatherRows, ctx, n, revIn, known, nKnown, seq, d, 1, b.twin(revIn),
                            b.hasTwin(revIn));
                    cell(b, s + "rev_", blk, revIn, revOut, nKnown, false);
                    cell(b, s, blk, n, yc, rows, false);
                    b.items(s + "cat", nKnown * seq * 2 * d);
                    g.task(s + "cat", TiRexKernels::concatForward, ctx, yc, revOut, known, big, nKnown, seq, d, b.twin(big),
                            b.hasTwin(big));
                    b.gemm(s + "comb", big, blk.combination(), cell, nKnown * seq, 0);
                    b.items(s + "scatter", nKnown * seq * d);
                    g.task(s + "scatter", TiRexKernels::scatterRows, ctx, cell, yc, known, nKnown, seq, d);
                    if (fuse) {
                        // h += yc and the FFN's RMSNorm in one pass
                        grid.addWorkerGrid(b.name + "." + s + "resid", group(tokens * TiRexKernels.NORM_THREADS, TiRexKernels.NORM_THREADS));
                        g.task(s + "resid", TiRexKernels::addRmsNorm, ctx, h, yc, n, w, blk.normFfn(), d, TiRex2Config.NORM_EPS,
                                b.twin(n), b.hasTwin(n));
                    } else {
                        b.items(s + "resid", tokens * d);
                        g.task(s + "resid", TiRexKernels::addInPlace, ctx, h, yc, tokens * d);
                        b.rms(s + "fnorm", h, n, blk.normFfn(), tokens, d, TiRex2Config.NORM_EPS);
                    }
                } else if (!cell(b, s, blk, n, h, rows, true, blk.normFfn())) {
                    b.rms(s + "fnorm", h, n, blk.normFfn(), tokens, d, TiRex2Config.NORM_EPS);
                }
                if (ffnUp != null) {
                    b.gemmToHalf(s + "up", n, blk.upGateUp(), ffnUp, tokens);
                    b.items(s + "swiglu", tokens * blk.ffnDim());
                    g.task(s + "swiglu", TiRexKernels::swiGluHalf, ctx, ffnUp, ffnSw, tokens, blk.ffnDim());
                    b.gemmFromHalf(s + "down", ffnSw, blk.down(), h, tokens, TiRexKernels.ACCUMULATE);
                } else {
                    b.gemm(s + "up", n, blk.upGateUp(), big, tokens, 0);
                    b.items(s + "swiglu", tokens * blk.ffnDim());
                    g.task(s + "swiglu", TiRexKernels::swiGlu, ctx, big, proj, tokens, blk.ffnDim(), b.twin(proj), b.hasTwin(proj));
                    b.gemm(s + "down", proj, blk.down(), h, tokens, TiRexKernels.ACCUMULATE);
                }
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

            this.grid = grid;
            exec = new TornadoExecutionPlan(b.graph);
            String device = System.getProperty("tirex.tornado.device");
            if (device != null && !device.isBlank()) {
                String[] dd = device.split(":");
                exec.withDevice(TornadoExecutionPlan.getDevice(Integer.parseInt(dd[0]), Integer.parseInt(dd[1])));
            }
            if (cudaGraph) {
                // CUDA backend only: the task graph is captured once and replayed as a CUDA graph
                exec.withCUDAGraph();
            }
            if (Boolean.getBoolean("tirex.tornado.profile")) {
                // per-task kernel times on stdout after every execution (analysis only; slows every call)
                exec.withProfiler(uk.ac.manchester.tornado.api.enums.ProfilerMode.CONSOLE);
            }
        }

        /**
         * Prints the activation buffers of this plan ({@code -Dtirex.tornado.memReport=true}): each
         * buffer's size, the FP16 twins, and the total per token. Each buffer has a host copy of the
         * same size; the weights are not included.
         */
        private void memoryReport(int tokens, Object[] named) {
            StringBuilder sb = new StringBuilder("[jitrex] plan " + shape + ", " + tokens + " tokens:");
            long total = 0;
            java.util.Set<Object> seen = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
            for (int i = 0; i < named.length; i += 2) {
                if (named[i + 1] instanceof uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray a && seen.add(a)) {
                    long bytes = a.getNumBytesOfSegment();
                    total += bytes;
                    sb.append(String.format(" %s=%.1fMB", named[i], bytes / 1e6));
                }
            }
            long twinBytes = 0;
            for (var e : twins.values()) {
                twinBytes += e.getNumBytesOfSegment();
            }
            total += twinBytes;
            sb.append(String.format(" twins=%.1fMB | total %.1f MB, %.1f KB per token", twinBytes / 1e6, total / 1e6,
                    total / 1024.0 / tokens));
            System.err.println(sb);
        }

        /**
         * One cell pass over {@code count} rows of {@code in} (already RMS-normalised) into
         * {@code dst}: the mLSTM (projection, cell, gated head norm, out projection) or the sLSTM
         * (gates, recurrence, head norm). With {@code add} the output is added to {@code dst},
         * otherwise stored. The mLSTM path overwrites {@code in} with the gated head norm.
         */
        private void cell(Builder b, String s, PackedWeights.Block blk, FloatArray in, FloatArray dst, int count,
                boolean add) {
            cell(b, s, blk, in, dst, count, add, -1);
        }

        /**
         * As above; with {@code ffnNorm >= 0} (and {@code add}) an sLSTM cell may also produce the
         * FFN's RMSNorm of the new residual stream into {@code n} (weight at {@code ffnNorm}),
         * fusing the per-head norm, the residual add and the RMSNorm. Returns whether it did.
         */
        private boolean cell(Builder b, String s, PackedWeights.Block blk, FloatArray in, FloatArray dst, int count,
                boolean add, int ffnNorm) {
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
                if (mlstmChunk(count, heads, seq)) {
                    int blocks = (seq + TiRexKernels.MLSTM_TB - 1) / TiRexKernels.MLSTM_TB;
                    b.grid.addWorkerGrid(b.name + "." + s + "mlstm", group(count * heads * blocks * (d / heads), d / heads));
                    b.g.task(s + "mlstm", TiRexKernels::mlstmChunk, b.ctx, proj, cell, count, seq, heads, pr.out(), qk, 2 * qk,
                            2 * qk + 2 * d, 2 * qk + 2 * d + heads, TiRex2Config.GATE_SOFT_CAP, TiRex2Config.MLSTM_EPS);
                } else {
                    b.grid.addWorkerGrid(b.name + "." + s + "mlstm", group(count * heads * seq * (d / heads), d / heads));
                    b.g.task(s + "mlstm", TiRexKernels::mlstmCell, b.ctx, proj, cell, count, seq, heads, pr.out(), qk, 2 * qk,
                            2 * qk + 2 * d, 2 * qk + 2 * d + heads, TiRex2Config.GATE_SOFT_CAP, TiRex2Config.MLSTM_EPS);
                }
                b.headNorm(s + "hnorm", cell, in, blk.headNorm(), tokens, heads, d / heads, proj, pr.out(), 2 * qk + d, 1);
                b.gemm(s + "out", in, blk.outProj(), dst, tokens, add ? TiRexKernels.ACCUMULATE : 0);
            } else {
                int heads = c.slstmHeads();
                int dh = d / heads;
                b.gemm(s + "gates", in, blk.slstmGates(), big, tokens, 0);
                int gpt = slstmGatesPerThread(dh);
                int threads = 4 * dh / gpt;
                int rb = slstmRowsPerGroup(count, heads);
                boolean inRegisters = halfRecurrent && registerRecurrent(dh, rb);
                if (inRegisters) {
                    rb = Math.min(rb, SLSTM_REG_MAX_ROWS);
                }
                int groups = (count + rb - 1) / rb * heads;
                b.grid.addWorkerGrid(b.name + "." + s + "slstm", group(groups * threads, threads));
                if (halfRecurrent) {
                    uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray rh = weights.recurrentHalf();
                    if (b.graphWeights.add(rh)) {
                        b.g.transferToDevice(DataTransferMode.FIRST_EXECUTION, rh);
                    }
                    if (inRegisters) {
                        b.g.task(s + "slstm", TiRexKernels::slstmRecurrenceRegR, b.ctx, big, b.w, rh,
                                weights.recurrentHalfOffset(blk), blk.slstmBias(), cell, count, seq, heads, rb);
                    } else {
                        b.g.task(s + "slstm", TiRexKernels::slstmRecurrenceHalfR, b.ctx, big, b.w, rh,
                                weights.recurrentHalfOffset(blk), blk.slstmBias(), cell, count, seq, heads, gpt, rb);
                    }
                } else {
                    b.g.task(s + "slstm", TiRexKernels::slstmRecurrence, b.ctx, big, b.w, blk.recurrent(), blk.slstmBias(),
                            cell, count, seq, heads, gpt, rb);
                }
                if (fuse && add && ffnNorm >= 0 && fusedNorm(heads, dh)) {
                    b.grid.addWorkerGrid(b.name + "." + s + "hnorm", group(tokens * heads * dh, heads * dh));
                    b.g.task(s + "hnorm", TiRexKernels::headNormAddRmsNorm, b.ctx, cell, dst, n, b.w, blk.groupNorm(), ffnNorm,
                            heads, dh, TiRex2Config.NORM_EPS, TiRex2Config.NORM_EPS, b.twin(n), b.hasTwin(n));
                    return true;
                }
                b.headNorm(s + "hnorm", cell, dst, blk.groupNorm(), tokens, heads, dh, proj, 0, 0, add ? 2 : 0);
            }
            return false;
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
            b.g.task(s + "vgather", TiRexKernels::gatherRows, b.ctx, h, xk, kept, nk, seq, d, 0, b.noTwin, 0);
            b.rms(s + "vnorm", xk, n, v.normAttn(), tokens, d, TiRex2Config.NORM_EPS);
            b.gemm(s + "qkv", n, v.qkv(), proj, tokens, 0);
            if (c.useQkNorm()) {
                int qkVectors = qkNormVectorsPerPass(heads, dh);
                b.grid.addWorkerGrid(b.name + "." + s + "qknorm", group(tokens * qkVectors * dh, qkVectors * dh));
                b.g.task(s + "qknorm", TiRexKernels::qkNorm, b.ctx, proj, tokens, heads, dh, 3 * d, Math.ulp(1.0f), qkVectors);
            }
            if (dh != 128) {
                throw new IllegalStateException("variateAttention assumes a head dim of 128");
            }
            int queryBlocks = (nk + TiRexKernels.ATTN_QUERIES - 1) / TiRexKernels.ATTN_QUERIES;
            b.grid.addWorkerGrid(b.name + "." + s + "attn", group(queryBlocks * seq * heads * 128, 128));
            b.g.task(s + "attn", TiRexKernels::variateAttention, b.ctx, proj, cell, groupStart, groupEnd, keptTarget, nk, seq,
                    heads, dh, 3 * d, b.twin(cell), b.hasTwin(cell));
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
            exec.withGridScheduler(grid).execute();
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
     * Builds the plan's task graph stage by stage ({@link #begin}, {@link #end}), and adds the recurring
     * task shapes (GEMM, norms, residual blocks) with their worker grids.
     */
    private final class Builder {
        final GridScheduler grid;
        final KernelContext ctx;
        final FloatArray w;
        uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray cHalf, noTwin;
        FloatArray noOut;
        java.util.Map<FloatArray, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray> twins;

        /** The FP16 twin of {@code f}, or the unused placeholder when it has none. */
        uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray twin(FloatArray f) {
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray t = twins.get(f);
            return t == null ? noTwin : t;
        }

        int hasTwin(FloatArray f) {
            return twins.containsKey(f) ? 1 : 0;
        }
        /** The task graph every stage adds to, and its snapshot once the last stage ends. */
        TaskGraph g;
        ImmutableTaskGraph graph;
        final String name = G;

        Builder(GridScheduler grid, KernelContext ctx, FloatArray w) {
            this.grid = grid;
            this.ctx = ctx;
            this.w = w;
        }

        final java.util.Set<Object> graphWeights = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());

        /** Starts a stage (the input stage, a block, the output stage) of the plan's one task graph. */
        TaskGraph begin(String stage) {
            if (g == null) {
                g = new TaskGraph(name);
            }
            return g;
        }

        /** Ends a stage; after the last one the graph is complete. */
        void end(boolean last) {
            if (last) {
                graph = g.snapshot();
            }
        }

        /** The worker grid of an element-wise kernel: one thread per item, in groups of {@link #ITEM_GROUP}. */
        void items(String task, int count) {
            grid.addWorkerGrid(this.name + "." + task, group((count + ITEM_GROUP - 1) / ITEM_GROUP * ITEM_GROUP, ITEM_GROUP));
        }

        void gemm(String name, FloatArray a, PackedWeights.Lin l, FloatArray c, int m, int flags) {
            if (gemmMode.cublas()) {
                cublasGemm(name, a, null, l, c, m, flags);
                return;
            }
            if (gemmMode == GemmMode.HALF || gemmMode == GemmMode.CUTILE) {
                tornadoHalfGemm(name, a, null, l, c, m, flags);
                return;
            }
            int tile = TiRexKernels.TILE;
            WorkerGrid wg = new WorkerGrid2D(ceil(l.out(), tile), ceil(m, tile));
            wg.setLocalWork(tile, tile, 1);
            grid.addWorkerGrid(this.name + "." + name, wg);
            int f = flags | (l.b() >= 0 ? TiRexKernels.BIAS : 0);
            g.task(name, TiRexKernels::gemm, ctx, a, 0, w, l.w(), Math.max(l.b(), 0), c, 0, m, l.out(), l.in(), f);
        }

        /** {@code H = A·Wᵀ} into an FP16-only buffer (FP32 accumulation, no bias or activation). */
        void gemmToHalf(String name, FloatArray a, PackedWeights.Lin l, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray out, int m) {
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray wh = halfWeight(l);
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray ah = halfInput(name, a, m, l.in());
            int n = l.out(), k = l.in();
            if (gemmMode.cublas()) {
                // the padding rows of the FP16 input hold finite stale values; their outputs are never read
                g.libraryTask(name, uk.ac.manchester.tornado.cublas.CuBlas::cublasGemmExFP16, 1, 0, n, pad(m), k, 1f, wh, k, ah, k,
                        0f, out, n);
                return;
            }
            int t = TiRexKernels.HGEMM_TILE, threads = TiRexKernels.HGEMM_THREADS;
            WorkerGrid wg = new WorkerGrid2D((n + t - 1) / t * threads, (m + t - 1) / t * threads);
            wg.setLocalWork(threads, threads, 1);
            g.task(name, TiRexKernels::gemmHalf, ctx, ah, wh, noOut, w, 0, m, n, k, TiRexKernels.TWIN_ONLY, out, 1);
            grid.addWorkerGrid(this.name + "." + name, wg);
        }

        /** {@code C = act(A·Wᵀ + b) (+ C)} from an FP16-only A, such as the FFN's SwiGLU output. */
        void gemmFromHalf(String name, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray ah, PackedWeights.Lin l, FloatArray c, int m, int flags) {
            if (gemmMode.cublas()) {
                cublasGemm(name, null, ah, l, c, m, flags);
            } else {
                tornadoHalfGemm(name, null, ah, l, c, m, flags);
            }
        }

        /**
         * {@code C = act(A·Wᵀ + b) (+ C)} with a TornadoVM FP16 GEMM: FP16 A (the producer's twin)
         * and W, FP32 accumulation and C, with the bias, SiLU, residual and C's own twin in the
         * kernel's epilogue, so no other task is needed.
         */
        private void tornadoHalfGemm(String name, FloatArray a, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray a16, PackedWeights.Lin l, FloatArray c, int m,
                int flags) {
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray wh = halfWeight(l);
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray ah = a16 != null ? a16 : halfInput(name, a, m, l.in());
            int n = l.out(), k = l.in();
            boolean bias = l.b() >= 0;
            int bOff = Math.max(l.b(), 0);
            WorkerGrid wg;
            if (gemmMode == GemmMode.CUTILE) {
                // the grid counts tile blocks; the block shape is the tile compiler's
                wg = new WorkerGrid2D((m + TiRexKernels.CT_M - 1) / TiRexKernels.CT_M, (n + TiRexKernels.CT_N - 1) / TiRexKernels.CT_N);
                g.task(name, TiRexKernels::gemmHalfTile, new uk.ac.manchester.tornado.api.tile.TileContext(), ah, wh, c, w, bOff, m, n,
                        k, bias ? 1f : 0f, (flags & TiRexKernels.SILU) != 0 ? 1f : 0f, (flags & TiRexKernels.ACCUMULATE) != 0 ? 1f : 0f,
                        twin(c), hasTwin(c));
            } else {
                int t = TiRexKernels.HGEMM_TILE, threads = TiRexKernels.HGEMM_THREADS;
                wg = new WorkerGrid2D((n + t - 1) / t * threads, (m + t - 1) / t * threads);
                wg.setLocalWork(threads, threads, 1);
                int f = flags | (bias ? TiRexKernels.BIAS : 0);
                g.task(name, TiRexKernels::gemmHalf, ctx, ah, wh, c, w, bOff, m, n, k, f, twin(c), hasTwin(c));
            }
            grid.addWorkerGrid(this.name + "." + name, wg);
        }

        /**
         * {@code C = act(A·Wᵀ + b) (+ C)} with cuBLAS. Row-major {@code C[m, n] = A[m, k]·W[n, k]ᵀ}
         * is the column-major {@code Cᵀ = op_T(W_cm)·A_cm}, i.e. {@code sgemm(T, N, n, m, k, W, k, A,
         * k, C, n)}. The bias is written into C first (then β = 1); SiLU runs afterwards.
         */
        private void cublasGemm(String name, FloatArray a, uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray a16, PackedWeights.Lin l, FloatArray c, int m, int flags) {
            boolean half = halfGemm(m);
            if (a16 != null && !half) {
                throw new IllegalStateException("GEMM " + name + " has only an FP16 input but runs in FP32");
            }
            boolean bias = l.b() >= 0;
            boolean acc = (flags & TiRexKernels.ACCUMULATE) != 0;
            int n = l.out(), k = l.in();
            if (half && fp16Kernel != Fp16Kernel.GEMMEX_MIXED) {
                // FP16 A, B and C on tensor cores, then one kernel back to FP32 with bias, SiLU and
                // the residual add fused (the residual stream stays FP32)
                uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray wh = halfWeight(l);
                uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray ah = a16 != null ? a16 : halfInput(name, a, m, k);
                if (fp16Kernel == Fp16Kernel.LT_H16) {
                    g.libraryTask(name, uk.ac.manchester.tornado.cublas.CuBlasLt::ltMatmulFP16, 1, 0, n, pad(m), k, 1f, wh, k,
                            ah, k, 0f, cHalf, n);
                } else {
                    g.libraryTask(name, uk.ac.manchester.tornado.cublas.CuBlas::cublasGemmExFP16, 1, 0, n, pad(m), k, 1f, wh, k,
                            ah, k, 0f, cHalf, n);
                }
                int f = (bias ? TiRexKernels.BIAS : 0) | (flags & (TiRexKernels.ACCUMULATE | TiRexKernels.SILU));
                items(name + "_epi", m * n);
                g.task(name + "_epi", TiRexKernels::fromHalfEpilogue, ctx, cHalf, c, w, Math.max(l.b(), 0), m, n, f, twin(c),
                        hasTwin(c));
                return;
            }
            FloatArray wm = half ? null : weights.matrix(l);
            if (wm != null && graphWeights.add(wm)) {
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, wm);
            }
            // SiLU without accumulation: the GEMM writes C (β = 0) and one epilogue adds the bias,
            // applies SiLU and writes the FP16 twin
            boolean siluEpilogue = fuse && (flags & TiRexKernels.SILU) != 0 && !acc;
            if (bias && !siluEpilogue) {
                items(name + "_bias", m * l.out());
                g.task(name + "_bias", TiRexKernels::biasFill, ctx, c, w, l.b(), m, l.out(), acc ? 1 : 0);
            }
            float beta = (bias && !siluEpilogue) || acc ? 1f : 0f;
            if (half) {
                uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray wh = halfWeight(l);
                uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray ah = a16 != null ? a16 : halfInput(name, a, m, k);
                // the row count is padded to a multiple of 8; the padding rows of the FP16 input
                // hold finite stale values and their outputs are never read
                g.libraryTask(name, uk.ac.manchester.tornado.cublas.CuBlas::cublasGemmExFP16FP32, 1, 0, n, pad(m), k, 1f, wh,
                        k, ah, k, beta, c, n);
            } else if (gemmMode == GemmMode.CUBLAS_TF32) {
                g.libraryTask(name, uk.ac.manchester.tornado.cublas.CuBlas::cublasSgemmTF32, 1, 0, n, m, k, 1f, wm, k, a, k,
                        beta, c, n);
            } else {
                g.libraryTask(name, uk.ac.manchester.tornado.cublas.CuBlas::cublasSgemm, 1, 0, n, m, k, 1f, wm, k, a, k,
                        beta, c, n);
            }
            if (siluEpilogue) {
                items(name + "_silu", m * n);
                g.task(name + "_silu", TiRexKernels::biasSilu, ctx, c, w, bias ? l.b() : -1, m, n, twin(c), hasTwin(c));
            } else if ((flags & TiRexKernels.SILU) != 0) {
                items(name + "_silu", m * n);
                g.task(name + "_silu", TiRexKernels::siluInPlace, ctx, c, m * n, twin(c), hasTwin(c));
            }
        }

        /** The FP16 A operand of a GEMM: {@code a}'s twin, which its producer has just written. */
        private uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray halfInput(String name, FloatArray a, int m, int k) {
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray t = twins.get(a);
            if (t == null) {
                throw new IllegalStateException("GEMM " + name + " reads a buffer without an FP16 twin");
            }
            return t;
        }

        private uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray halfWeight(PackedWeights.Lin l) {
            uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray wh = weights.halfMatrix(l);
            if (graphWeights.add(wh)) {
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, wh);
            }
            return wh;
        }

        void rms(String name, FloatArray x, FloatArray y, int wOff, int rows, int d, float eps) {
            grid.addWorkerGrid(this.name + "." + name, group(rows * TiRexKernels.NORM_THREADS, TiRexKernels.NORM_THREADS));
            g.task(name, TiRexKernels::rmsNorm, ctx, x, y, w, wOff, d, eps, twin(y), hasTwin(y));
        }

        void headNorm(String name, FloatArray x, FloatArray y, int wOff, int tokens, int heads, int dh, FloatArray gate,
                int gStride, int gOff, int mode) {
            grid.addWorkerGrid(this.name + "." + name, group(tokens * heads * dh, dh));
            // mode 2 adds into the residual stream, which no GEMM reads directly: no twin
            int tw = mode == 2 ? 0 : hasTwin(y);
            g.task(name, TiRexKernels::headLayerNorm, ctx, x, y, w, wOff, heads, dh, TiRex2Config.NORM_EPS, gate,
                    gStride, gOff, mode, tw == 1 ? twin(y) : noTwin, tw);
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
