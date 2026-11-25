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
import ai.beestream.jitcast.toto.TotoPipeline;
import ai.beestream.jitcast.toto.TotoWeights;
import ai.beestream.jitcast.toto.cpu.CpuToto;
import ai.beestream.jitcast.toto.tornado.TornadoTotoForward;
import ai.beestream.jitcast.toto.tornado.TotoPacked;
import ai.beestream.jitcast.util.Json;

/**
 * End-to-end Toto 2.0 forecast latency on the scenarios of a workload file, the same inputs the reference
 * engine is timed on. Each scenario forecasts with the default settings, a context cap covering its whole
 * series and one forward pass for the whole batch; a scenario's past covariates are extra variates, and its
 * known-future covariates are left out, since Toto 2.0 takes none:
 *
 * <pre>
 * java ... ai.beestream.jitcast.bench.TotoBench --checkpoint toto-2.0-22m --workloads workloads.safetensors \
 *     [--engine tornado|cpu] [--scenarios uni_b1,uni_b8] [--warmup 3] [--reps 20] [--budget 5] [--out tornado.json]
 * </pre>
 */
public final class TotoBench {

    private TotoBench() {}

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
        try (TotoWeights weights = TotoWeights.load(Path.of(a.get("checkpoint")));
                SafeTensors w = SafeTensors.open(Path.of(a.get("workloads")))) {
            TornadoTotoForward tornado = engine.equals("cpu") ? null : new TornadoTotoForward(weights.config(), TotoPacked.pack(weights));
            try {
                TotoPipeline base = new TotoPipeline(weights.config(),
                        tornado != null ? tornado : new CpuToto(weights));
                results.put("engine", "toto-" + engine);
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
                    List<TimeSeries> series = series(w.get(name + ".context"), w.contains(name + ".past") ? w.get(name + ".past") : null);
                    int length = series.get(0).length(), p32 = weights.config().patchSize();
                    // the whole context: the limit covers it plus the horizon's patches but one
                    int maxContext = length + (Math.ceilDiv(h, p32) - 1) * p32;
                    TotoPipeline p = base.withOptions(TotoPipeline.Options.defaults().withMaxContext(maxContext))
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
                    r.put("past", series.get(0).pastCount());
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

    /** One series per batch row of a {@code (B, V, T)} context, with its {@code (B, V_p, T)} past covariates if any. */
    private static List<TimeSeries> series(Tensor ctx, Tensor past) {
        List<TimeSeries> out = new ArrayList<>();
        for (int i = 0; i < ctx.dim(0); i++) {
            out.add(new TimeSeries(rows(ctx, i), past == null ? null : rows(past, i), null));
        }
        return out;
    }

    private static float[][] rows(Tensor t, int i) {
        int v = t.dim(1), n = t.dim(2);
        float[] flat = t.toFloatArray();
        float[][] rows = new float[v][];
        for (int j = 0; j < v; j++) {
            rows[j] = Arrays.copyOfRange(flat, (i * v + j) * n, (i * v + j + 1) * n);
        }
        return rows;
    }
}
