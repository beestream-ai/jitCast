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

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.t0.T0Config;
import ai.beestream.jitcast.t0.T0Forward;
import ai.beestream.jitcast.t0.T0Pipeline;
import ai.beestream.jitcast.t0.T0Weights;
import ai.beestream.jitcast.t0.cpu.CpuT0;
import ai.beestream.jitcast.t0.tornado.T0Packed;
import ai.beestream.jitcast.t0.tornado.TornadoT0Forward;
import ai.beestream.jitcast.util.Json;

/**
 * End-to-end t0 forecast latency on the scenarios of a workload file, the same inputs the reference engine
 * is timed on. Each batch row is one series with its past and known-future covariates; each scenario
 * forecasts the checkpoint's trained levels with the default options and the whole batch in one pass:
 *
 * <pre>
 * java ... ai.beestream.jitcast.bench.T0Bench --checkpoint t0-gamma --workloads workloads.safetensors \
 *     [--engine tornado|cpu] [--scenarios uni_b1,cov_b8] [--warmup 3] [--reps 20] [--budget 5] [--out tornado.json]
 * </pre>
 */
public final class T0Bench {

    private T0Bench() {}

    public static void main(String[] argv) throws IOException {
        Map<String, String> a = new HashMap<>();
        for (int i = 0; i + 1 < argv.length; i += 2) {
            a.put(argv[i].replaceFirst("^--", ""), argv[i + 1]);
        }
        String engine = a.getOrDefault("engine", "tornado");
        int warmup = Integer.parseInt(a.getOrDefault("warmup", "3"));
        int reps = Integer.parseInt(a.getOrDefault("reps", "20"));
        double budget = Double.parseDouble(a.getOrDefault("budget", "5"));
        List<String> wanted = a.containsKey("scenarios") ? List.of(a.get("scenarios").split(",")) : null;

        Map<String, Object> results = new LinkedHashMap<>();
        try (T0Weights weights = T0Weights.load(Path.of(a.get("checkpoint")));
                SafeTensors w = SafeTensors.open(Path.of(a.get("workloads")))) {
            T0Config c = weights.config();
            TornadoT0Forward tornado = engine.equals("cpu") ? null : new TornadoT0Forward(c, T0Packed.pack(weights));
            try {
                T0Forward forward = tornado != null ? tornado : new CpuT0(weights);
                results.put("engine", "t0-" + engine);
                if (tornado != null) {
                    results.put("gemm", tornado.gemmMode().name().toLowerCase().replace('_', '-'));
                }
                Map<String, Object> scenarios = new LinkedHashMap<>();
                for (Map.Entry<String, String> s : w.metadata().entrySet()) {
                    String name = s.getKey();
                    if (name.equals("format") || wanted != null && !wanted.contains(name)) {
                        continue;
                    }
                    int h = Integer.parseInt(s.getValue());
                    List<TimeSeries> series = series(w.get(name + ".context"),
                            w.contains(name + ".past") ? w.get(name + ".past") : null,
                            w.contains(name + ".known") ? w.get(name + ".known") : null);
                    int length = series.get(0).length();
                    T0Pipeline p = new T0Pipeline(c, forward, T0Pipeline.Options.defaults(c), series.size(), Long.MAX_VALUE);
                    long t0 = System.nanoTime();
                    List<Forecast> first = p.forecast(series, h);
                    double firstMs = (System.nanoTime() - t0) / 1e6;
                    for (int i = 0; i < warmup; i++) {
                        p.forecast(series, h);
                    }
                    List<Double> ms = new ArrayList<>();
                    long deadline = System.nanoTime() + (long) (budget * 1e9);
                    while (ms.size() < reps && (ms.size() < 3 || System.nanoTime() < deadline)) {
                        t0 = System.nanoTime();
                        p.forecast(series, h);
                        ms.add((System.nanoTime() - t0) / 1e6);
                    }
                    double[] sorted = ms.stream().mapToDouble(Double::doubleValue).sorted().toArray();
                    double median = sorted.length % 2 == 1 ? sorted[sorted.length / 2]
                            : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
                    Map<String, Object> r = new LinkedHashMap<>();
                    r.put("H", h);
                    r.put("batch", series.size());
                    r.put("context", length);
                    r.put("median_ms", median);
                    r.put("min_ms", sorted[0]);
                    r.put("first_ms", firstMs);
                    r.put("reps", sorted.length);
                    r.put("median0", first.get(0).median(0)[0]);
                    scenarios.put(name, r);
                    System.out.printf("%-18s median %9.2f ms  min %9.2f ms  first %9.1f ms%n", name, median, sorted[0], firstMs);
                }
                results.put("scenarios", scenarios);
            } finally {
                if (tornado != null) {
                    tornado.close();
                }
            }
        }
        if (a.containsKey("out")) {
            Files.writeString(Path.of(a.get("out")), Json.write(results));
        }
    }

    /** One series per batch row: targets {@code (B, V, T)}, past covariates {@code (B, C, T)}, known {@code (B, C, T + H)}. */
    private static List<TimeSeries> series(Tensor ctx, Tensor past, Tensor known) {
        int b = ctx.dim(0);
        float[][][] targets = rows(ctx), p = past != null ? rows(past) : null, k = known != null ? rows(known) : null;
        List<TimeSeries> out = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            out.add(new TimeSeries(targets[i], p != null ? p[i] : null, k != null ? k[i] : null));
        }
        return out;
    }

    private static float[][][] rows(Tensor t) {
        int b = t.dim(0), v = t.dim(1), n = t.dim(2);
        float[] flat = t.toFloatArray();
        float[][][] out = new float[b][v][];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < v; j++) {
                out[i][j] = Arrays.copyOfRange(flat, (i * v + j) * n, (i * v + j + 1) * n);
            }
        }
        return out;
    }
}
