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
import ai.beestream.jitcast.timesfm.TimesFmPipeline;
import ai.beestream.jitcast.timesfm.TimesFmWeights;
import ai.beestream.jitcast.timesfm.cpu.CpuTimesFm;
import ai.beestream.jitcast.timesfm.tornado.TimesFmForward;
import ai.beestream.jitcast.timesfm.tornado.TimesFmPacked;
import ai.beestream.jitcast.util.Json;

/**
 * End-to-end TimesFM 2.5 forecast latency on the univariate scenarios of a workload file, the same inputs
 * the reference engine is timed on. Each scenario forecasts with the recommended settings, a context cap
 * equal to its series' length and one forward pass for the whole batch:
 *
 * <pre>
 * java ... ai.beestream.jitcast.bench.TimesFmBench --checkpoint timesfm-2.5-200m --workloads workloads.safetensors \
 *     [--engine tornado|cpu] [--scenarios uni_b1,uni_b8] [--warmup 3] [--reps 20] [--budget 5] [--out tornado.json]
 * </pre>
 */
public final class TimesFmBench {

    private TimesFmBench() {}

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
        try (TimesFmWeights weights = TimesFmWeights.load(Path.of(a.get("checkpoint")));
                SafeTensors w = SafeTensors.open(Path.of(a.get("workloads")))) {
            TimesFmForward tornado = engine.equals("cpu") ? null : new TimesFmForward(weights.config(), TimesFmPacked.pack(weights));
            try {
                TimesFmPipeline base = new TimesFmPipeline(weights.config(),
                        tornado != null ? tornado : new CpuTimesFm(weights));
                results.put("engine", "timesfm-" + engine);
                if (tornado != null) {
                    results.put("gemm", tornado.gemmMode().name().toLowerCase().replace('_', '-'));
                }
                Map<String, Object> scenarios = new LinkedHashMap<>();
                for (Map.Entry<String, String> s : w.metadata().entrySet()) {
                    String name = s.getKey();
                    if (name.equals("format") || w.contains(name + ".past") || w.contains(name + ".known")
                            || wanted != null && !wanted.contains(name)) {
                        continue;
                    }
                    int h = Integer.parseInt(s.getValue());
                    List<TimeSeries> series = series(w.get(name + ".context"));
                    int length = series.get(0).length();
                    TimesFmPipeline p = base.withOptions(TimesFmPipeline.Options.defaults(weights.config()).withMaxContext(length))
                            .withBatchSize(series.size());
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

    /** One univariate series per batch row of a {@code (B, 1, T)} context. */
    private static List<TimeSeries> series(Tensor ctx) {
        int b = ctx.dim(0), v = ctx.dim(1), n = ctx.dim(2);
        float[] flat = ctx.toFloatArray();
        List<TimeSeries> out = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            float[][] rows = new float[v][];
            for (int j = 0; j < v; j++) {
                rows[j] = Arrays.copyOfRange(flat, (i * v + j) * n, (i * v + j + 1) * n);
            }
            out.add(new TimeSeries(rows, null, null));
        }
        return out;
    }
}
