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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.ForecastEngine;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.chronos2.cpu.CpuChronos2Engine;
import ai.beestream.jitcast.chronos2.tornado.TornadoChronos2Engine;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.util.Json;

/**
 * End-to-end Chronos-2 forecast latency on the scenarios {@code scripts/chronos2/bench_reference.py
 * workloads} writes, the same inputs the reference is timed on:
 *
 * <pre>
 * java ... ai.beestream.jitcast.bench.Chronos2Bench --checkpoint chronos-2-small --workloads workloads.safetensors \
 *     [--engine tornado|cpu] [--scenarios uni_b1,uni_b8] [--warmup 3] [--reps 20] [--budget 5] [--out tornado.json]
 * </pre>
 */
public final class Chronos2Bench {

    private Chronos2Bench() {}

    public static void main(String[] argv) throws IOException {
        Map<String, String> a = new HashMap<>();
        for (int i = 0; i + 1 < argv.length; i += 2) {
            a.put(argv[i].replaceFirst("^--", ""), argv[i + 1]);
        }
        Path checkpoint = Path.of(a.get("checkpoint"));
        String engineName = a.getOrDefault("engine", "tornado");
        int warmup = Integer.parseInt(a.getOrDefault("warmup", "3"));
        int reps = Integer.parseInt(a.getOrDefault("reps", "20"));
        double budget = Double.parseDouble(a.getOrDefault("budget", "5"));
        List<String> wanted = a.containsKey("scenarios") ? List.of(a.get("scenarios").split(",")) : null;

        Map<String, Object> results = new LinkedHashMap<>();
        try (ForecastEngine engine = engineName.equals("cpu") ? CpuChronos2Engine.load(checkpoint)
                : TornadoChronos2Engine.load(checkpoint);
                SafeTensors w = SafeTensors.open(Path.of(a.get("workloads")))) {
            results.put("engine", engine.name());
            Map<String, Object> scenarios = new LinkedHashMap<>();
            for (Map.Entry<String, String> s : w.metadata().entrySet()) {
                String name = s.getKey();
                if (name.equals("format") || wanted != null && !wanted.contains(name)) {
                    continue;
                }
                int h = Integer.parseInt(s.getValue());
                List<TimeSeries> series = series(w, name);
                long t0 = System.nanoTime();
                engine.forecast(series, h);
                double first = (System.nanoTime() - t0) / 1e6;
                for (int i = 0; i < warmup; i++) {
                    engine.forecast(series, h);
                }
                List<Double> ms = new ArrayList<>();
                long deadline = System.nanoTime() + (long) (budget * 1e9);
                while (ms.size() < reps && (ms.size() < 3 || System.nanoTime() < deadline)) {
                    t0 = System.nanoTime();
                    engine.forecast(series, h);
                    ms.add((System.nanoTime() - t0) / 1e6);
                }
                double[] sorted = ms.stream().mapToDouble(Double::doubleValue).sorted().toArray();
                double median = sorted.length % 2 == 1 ? sorted[sorted.length / 2]
                        : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("H", h);
                r.put("batch", series.size());
                r.put("median_ms", median);
                r.put("min_ms", sorted[0]);
                r.put("first_ms", first);
                r.put("reps", sorted.length);
                scenarios.put(name, r);
                System.out.printf("%-18s median %9.2f ms  min %9.2f ms  first %9.1f ms%n", name, median, sorted[0], first);
            }
            results.put("scenarios", scenarios);
        }
        if (a.containsKey("out")) {
            Files.writeString(Path.of(a.get("out")), Json.write(results));
        }
    }

    private static List<TimeSeries> series(SafeTensors w, String name) {
        float[][][] ctx = cube(w.get(name + ".context"));
        float[][][] past = w.contains(name + ".past") ? cube(w.get(name + ".past")) : null;
        float[][][] known = w.contains(name + ".known") ? cube(w.get(name + ".known")) : null;
        List<TimeSeries> out = new ArrayList<>();
        for (int b = 0; b < ctx.length; b++) {
            out.add(new TimeSeries(ctx[b], past == null ? null : past[b], known == null ? null : known[b]));
        }
        return out;
    }

    private static float[][][] cube(Tensor t) {
        int a = t.dim(0), b = t.dim(1), n = t.dim(2);
        float[] flat = t.toFloatArray();
        float[][][] r = new float[a][b][];
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                r[i][j] = Arrays.copyOfRange(flat, (i * b + j) * n, (i * b + j + 1) * n);
            }
        }
        return r;
    }
}
