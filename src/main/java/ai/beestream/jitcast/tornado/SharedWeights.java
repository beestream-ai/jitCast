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
package ai.beestream.jitcast.tornado;

import java.util.ArrayList;
import java.util.Collection;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * One device copy of a model's weights, shared by every compiled plan of an engine. Weight buffers are
 * uploaded by producer plans of their own, which persist them; each shape's plan consumes them from
 * there ({@code TaskGraph.consumeFromDevice(ImmutableTaskGraph, …)}, TornadoVM 8.2.0) instead of
 * uploading its own copy. The first plan that needs a buffer adds it, in one producer plan for all the
 * buffers that plan needs; later plans find them in place. The producers stay open until {@link #close()},
 * so close this after the plans that consume from it.
 */
public final class SharedWeights implements AutoCloseable {

    /** Work-group width of the producers' tasks. */
    private static final int GROUP = 32;

    private final String name;
    private final TornadoDevice device;
    private final Map<Object, ImmutableTaskGraph> owner = new IdentityHashMap<>();
    private final List<TornadoExecutionPlan> producers = new ArrayList<>();
    private long bytes;

    /**
     * Shared weights on one device.
     *
     * @param name a prefix for the producer graphs' names, unique among the engine's graphs
     * @param device the device the consuming plans run on
     */
    public SharedWeights(String name, TornadoDevice device) {
        this.name = name;
        this.device = device;
    }

    /** Reads element 0 of a weight buffer, so that its producer graph allocates and keeps it. */
    public static void keep(KernelContext ctx, FloatArray a, IntArray flag) {
        int i = ctx.globalIdx;
        if (i < 1) {
            flag.set(0, a.get(0) == a.get(0) ? 1 : 0);
        }
    }

    /** {@link #keep} for an FP16 buffer. */
    public static void keepHalf(KernelContext ctx, HalfFloatArray a, IntArray flag) {
        int i = ctx.globalIdx;
        if (i < 1) {
            float v = a.get(0).getFloat32();
            flag.set(0, v == v ? 1 : 0);
        }
    }

    /**
     * Declares on {@code g} that it consumes {@code buffers} from the shared copy, uploading those not on
     * the device yet in one new producer plan first. Call it once the graph's tasks are added.
     *
     * @param g the consuming graph
     * @param buffers its weight buffers, {@link FloatArray}s and {@link HalfFloatArray}s
     */
    public synchronized void consume(TaskGraph g, Collection<?> buffers) {
        List<Object> missing = new ArrayList<>();
        for (Object b : buffers) {
            if (!owner.containsKey(b) && !missing.contains(b)) {
                missing.add(b);
            }
        }
        if (!missing.isEmpty()) {
            upload(missing);
        }
        for (Object b : buffers) {
            g.consumeFromDevice(owner.get(b), b);
        }
    }

    private void upload(List<Object> buffers) {
        String graphName = name + "_w" + producers.size();
        IntArray flag = new IntArray(1);
        TaskGraph g = new TaskGraph(graphName);
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, buffers.toArray());
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, flag);
        GridScheduler grid = new GridScheduler();
        KernelContext ctx = new KernelContext();
        for (int i = 0; i < buffers.size(); i++) {
            Object b = buffers.get(i);
            String task = "k" + i;
            if (b instanceof HalfFloatArray h) {
                g.task(task, SharedWeights::keepHalf, ctx, h, flag);
            } else if (b instanceof FloatArray f) {
                g.task(task, SharedWeights::keep, ctx, f, flag);
            } else {
                throw new IllegalArgumentException("unsupported weight buffer " + b.getClass().getSimpleName());
            }
            WorkerGrid wg = new WorkerGrid1D(GROUP);
            wg.setLocalWork(GROUP, 1, 1);
            grid.addWorkerGrid(graphName + "." + task, wg);
        }
        g.persistOnDevice(buffers.toArray());
        ImmutableTaskGraph snapshot = g.snapshot();
        TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot);
        plan.withGridScheduler(grid);
        plan.withDevice(device);
        plan.execute();
        producers.add(plan);
        for (Object b : buffers) {
            owner.put(b, snapshot);
            bytes += size(b);
        }
    }

    /**
     * {@return the device bytes of the given buffers}
     *
     * @param buffers weight buffers
     */
    public static long size(Collection<?> buffers) {
        long n = 0;
        for (Object b : buffers) {
            n += size(b);
        }
        return n;
    }

    private static long size(Object b) {
        return b instanceof TornadoNativeArray a ? a.getNumBytesOfSegment() : 0;
    }

    /**
     * {@return an estimate of a model's shared weights on the device}: the packed buffer, plus the
     * per-matrix FP32 copies the cuBLAS FP32 and TF32 GEMMs read, plus the FP16 copies of the FP16 GEMMs
     *
     * @param packedBytes the packed weight buffer's bytes
     * @param fp32Copies whether the GEMMs read per-matrix FP32 copies
     * @param fp16Copies whether the GEMMs read per-matrix FP16 copies
     */
    public static long estimate(long packedBytes, boolean fp32Copies, boolean fp16Copies) {
        return packedBytes + (fp32Copies ? packedBytes : 0) + (fp16Copies ? packedBytes / 2 : 0);
    }

    /** {@return the device bytes of every shared weight buffer so far} */
    public synchronized long deviceBytes() {
        return bytes;
    }

    @Override
    public synchronized void close() {
        RuntimeException first = null;
        for (TornadoExecutionPlan p : producers) {
            try {
                p.close();
            } catch (Exception e) {
                if (first == null) {
                    first = new IllegalStateException("closing a shared-weights plan", e);
                } else {
                    first.addSuppressed(e);
                }
            }
        }
        producers.clear();
        owner.clear();
        if (first != null) {
            throw first;
        }
    }
}
