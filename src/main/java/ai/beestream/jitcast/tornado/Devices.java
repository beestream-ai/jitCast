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

import ai.beestream.jitcast.util.Options;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/** The TornadoVM device jitcast runs on, its memory budget, and jitcast's TornadoVM defaults. */
public final class Devices {

    private Devices() {}

    /** TornadoVM's own default for {@code tornado.device.memory}. */
    private static final long TORNADO_DEFAULT_MEMORY = 4L << 30;

    /**
     * Sets jitcast's TornadoVM defaults, unless already set. TornadoVM reads them when its backends
     * start, so every engine calls this before its first TornadoVM call, and so does code that calls
     * TornadoVM before creating an engine. TornadoVM's Metal backend spin-waits up to 1 ms on each
     * dispatch before it blocks; a forward pass runs dozens of kernels per task graph, so its waits
     * outlast the spin and jitcast runs up to 2× slower on Metal. The spin is off.
     */
    public static void defaults() {
        if (System.getProperty("tornado.metal.dispatch.spinWait") == null) {
            System.setProperty("tornado.metal.dispatch.spinWait", "False");
        }
    }

    /**
     * The device of {@code -Djitcast.tornado.device=<backend>:<device>} (as {@code tornado --devices}
     * numbers them), or TornadoVM's default device.
     */
    public static TornadoDevice device() {
        defaults();
        String prop = System.getProperty("jitcast.tornado.device");
        if (prop == null || prop.isBlank()) {
            return TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
        }
        String[] dd = prop.trim().split(":");
        int backend, index;
        try {
            if (dd.length != 2) {
                throw new NumberFormatException();
            }
            backend = Integer.parseInt(dd[0].trim());
            index = Integer.parseInt(dd[1].trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("-Djitcast.tornado.device=" + prop + ": expected <backend>:<device>, such as 0:0");
        }
        var runtime = TornadoRuntimeProvider.getTornadoRuntime();
        if (backend < 0 || backend >= runtime.getNumBackends() || index < 0
                || index >= runtime.getBackend(backend).getNumDevices()) {
            throw new IllegalArgumentException("-Djitcast.tornado.device=" + prop + ": no such device; see tornado --devices");
        }
        return TornadoExecutionPlan.getDevice(backend, index);
    }

    /**
     * The device memory TornadoVM lets jitcast use: {@code tornado.device.memory}, TornadoVM's own
     * cap on the device buffers it allocates (4 GB unless set). The plan caches keep their plans
     * within it.
     */
    public static long memory() {
        String v = System.getProperty("tornado.device.memory");
        if (v == null || v.isBlank()) {
            return TORNADO_DEFAULT_MEMORY;
        }
        long b = Options.parseBytes(v);
        if (b <= 0) {
            throw new IllegalArgumentException("-Dtornado.device.memory=" + v + ": expected a size such as 16GB");
        }
        return b;
    }
}
