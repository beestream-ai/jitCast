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
package ai.beestream.jitcast.bench;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.tirex2.ForecastPipeline;
import ai.beestream.jitcast.tirex2.TiRex2Weights;
import ai.beestream.jitcast.tirex2.tornado.PackedWeights;
import ai.beestream.jitcast.tirex2.tornado.TornadoForward;
import ai.beestream.jitcast.util.Json;

/**
 * One cold-start measurement of the TornadoVM engine, the counterpart of
 * {@code bench_nxai.py coldstart}: a single series of 2048 steps, horizon 64. It appends one JSON
 * line with the checkpoints (all in ms):
 *
 * <ul>
 *   <li>{@code jvm_to_main}: JVM start to {@code main};
 *   <li>{@code runtime_init}: TornadoVM runtime and CUDA context (the default device);
 *   <li>{@code load}: weights mapped, bound and packed;
 *   <li>{@code first_forecast}: the first {@code forecast()}, including plan build, kernel
 *       compilation (or the code cache) and weight upload;
 *   <li>{@code second_forecast}: a warm call, for reference;
 *   <li>{@code exec_to_first_forecast}: process exec (from {@code BENCH_EXEC_START_NS}) to the end
 *       of the first forecast.
 * </ul>
 */
public final class TiRexColdStart {

    public static void main(String[] args) throws IOException {
        long tMain = System.currentTimeMillis();
        Map<String, String> a = TiRexBench.parse(args);
        long jvmStart = ManagementFactory.getRuntimeMXBean().getStartTime();

        TornadoForward.tornadoDefaults();
        long t0 = System.nanoTime();
        uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice().getDeviceName();
        long t1 = System.nanoTime();
        TiRex2Weights weights = TiRex2Weights.load(Path.of(a.get("weights")));
        TornadoForward forward = new TornadoForward(weights.config(), PackedWeights.pack(weights), 1);
        ForecastPipeline pipeline = new ForecastPipeline(weights.config(), forward);
        float[] series;
        try (SafeTensors w = SafeTensors.open(Path.of(a.get("workloads")))) {
            series = TiRexBench.rows(w.get("univariate"))[0];
        }
        long t2 = System.nanoTime();
        List<TimeSeries> in = List.of(TimeSeries.univariate(series));
        pipeline.forecast(in, 64);
        long t3 = System.nanoTime();
        long tFirstEpochNs = System.currentTimeMillis() * 1_000_000L;
        pipeline.forecast(in, 64);
        long t4 = System.nanoTime();

        Map<String, Object> r = new LinkedHashMap<>();
        r.put("engine", "tornado");
        r.put("config", a.getOrDefault("config", "tornado"));
        r.put("cache", a.getOrDefault("cache", "?"));
        r.put("jvm_to_main_ms", (double) (tMain - jvmStart));
        r.put("runtime_init_ms", (t1 - t0) / 1e6);
        r.put("load_ms", (t2 - t1) / 1e6);
        r.put("first_forecast_ms", (t3 - t2) / 1e6);
        r.put("second_forecast_ms", (t4 - t3) / 1e6);
        String exec = System.getenv("BENCH_EXEC_START_NS");
        if (exec != null && !exec.isBlank()) {
            r.put("exec_to_first_forecast_ms", (tFirstEpochNs - Long.parseLong(exec.trim())) / 1e6);
        }
        String line = Json.write(r);
        System.out.println(line);
        Files.writeString(Path.of(a.get("out")), line + "\n", StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        forward.close();
        weights.close();
        System.exit(0);
    }

    private TiRexColdStart() {}
}
