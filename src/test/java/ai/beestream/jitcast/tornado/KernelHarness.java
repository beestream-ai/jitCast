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

import static org.junit.Assert.assertTrue;

import java.util.function.Consumer;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Runs one kernel on its own and compares it with its CPU reference, for the kernel parity tests of
 * every family. The device is {@code -Djitcast.tornado.device=<driver>:<device>}, or the default one.
 */
public final class KernelHarness {

    public static final double TOLERANCE = 1e-4;

    /** The FP16 twin argument of a kernel called with {@code twin = 0}. */
    public static final HalfFloatArray NO_TWIN = new HalfFloatArray(1);

    private KernelHarness() {}

    /** The device the tests run on. */
    public static TornadoDevice device() {
        String device = System.getProperty("jitcast.tornado.device");
        if (device != null && !device.isBlank()) {
            String[] dd = device.split(":");
            return TornadoExecutionPlan.getDevice(Integer.parseInt(dd[0]), Integer.parseInt(dd[1]));
        }
        return TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
    }

    /** Runs one task graph that reads {@code inputs} and returns {@code output}. */
    public static void run(String task, WorkerGrid wg, Consumer<TaskGraph> addTask, FloatArray output, FloatArray... inputs) {
        TaskGraph g = new TaskGraph("k").transferToDevice(DataTransferMode.EVERY_EXECUTION, (Object[]) inputs);
        g.transferToDevice(DataTransferMode.EVERY_EXECUTION, output);
        addTask.accept(g);
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, output);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            if (wg != null) {
                GridScheduler grid = new GridScheduler();
                grid.addWorkerGrid("k." + task, wg);
                plan.withGridScheduler(grid);
            }
            plan.withDevice(device());
            plan.execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    /** Runs a task graph whose one task is {@code k.t}, with its transfers already declared. */
    public static void execute(TaskGraph g, WorkerGrid wg) {
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            GridScheduler grid = new GridScheduler();
            grid.addWorkerGrid("k.t", wg);
            plan.withGridScheduler(grid);
            plan.withDevice(device());
            plan.execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    /** One thread per item, 128 per group (the engine's element-wise launch). */
    public static WorkerGrid items(int count) {
        return grid1((count + 127) / 128 * 128, 128);
    }

    public static WorkerGrid grid1(int global, int local) {
        WorkerGrid wg = new WorkerGrid1D(global);
        wg.setLocalWork(local, 1, 1);
        return wg;
    }

    public static long maxWorkGroup() {
        return device().getDeviceMaxWorkgroupDimensions()[0];
    }

    /** Fails unless {@code actual} is within {@link #TOLERANCE} of {@code expected}, relative to its largest value. */
    public static void check(String what, float[] actual, float[] expected) {
        double diff = 0, scale = 1e-6;
        int nan = 0;
        for (int i = 0; i < expected.length; i++) {
            if (Float.isNaN(actual[i]) && !Float.isNaN(expected[i])) {
                nan++;
            }
            diff = Math.max(diff, Math.abs(actual[i] - expected[i]));
            scale = Math.max(scale, Math.abs(expected[i]));
        }
        double err = diff / scale;
        System.out.printf("%-16s rel max err %.2e  NaN %d%n", what, err, nan);
        assertTrue(what + ": " + err + ", NaN " + nan, nan == 0 && err < TOLERANCE);
    }

    public static HalfFloatArray half(float[] v) {
        HalfFloatArray h = new HalfFloatArray(v.length);
        for (int i = 0; i < v.length; i++) {
            h.getSegment().setAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, i, Float.floatToFloat16(v[i]));
        }
        return h;
    }

    public static float[] fromTwin(HalfFloatArray h, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = Float.float16ToFloat(h.getSegment().getAtIndex(java.lang.foreign.ValueLayout.JAVA_SHORT, i));
        }
        return v;
    }

    public static float[] roundHalf(float[] v) {
        float[] r = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            r[i] = Float.float16ToFloat(Float.floatToFloat16(v[i]));
        }
        return r;
    }
}
