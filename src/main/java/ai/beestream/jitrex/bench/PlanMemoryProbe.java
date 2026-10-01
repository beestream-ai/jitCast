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
package ai.beestream.jitrex.bench;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

import ai.beestream.jitrex.model.ModelBatch;
import ai.beestream.jitrex.model.TiRex2Weights;
import ai.beestream.jitrex.tornado.PackedWeights;
import ai.beestream.jitrex.tornado.TornadoForward;

/**
 * Builds and evicts plans of the given batch sizes and prints, after each forward pass, this
 * process's device memory (from {@code nvidia-smi}, 0 without an NVIDIA GPU) and the JVM's off-heap
 * memory after a GC, to show whether evicting and closing plans gives their memory back. Repeat a
 * small set of sizes (e.g. {@code --rows 64,72,80 --rounds 6}) so the live plans stay the same size.
 *
 * <pre>
 * java … PlanMemoryProbe --weights model.safetensors [--rows 8,16,24,…] [--rounds 2] [--plans 1]
 * </pre>
 */
public final class PlanMemoryProbe {

    public static void main(String[] args) throws IOException, InterruptedException {
        Map<String, String> a = TiRexBench.parse(args);
        TiRex2Weights weights = TiRex2Weights.load(Path.of(a.get("weights")));
        int[] sizes = Arrays.stream(a.getOrDefault("rows", "8,16,24,32,40,48,56,64").split(",")).mapToInt(Integer::parseInt)
                .toArray();
        int rounds = Integer.parseInt(a.getOrDefault("rounds", "2"));
        int cached = Integer.parseInt(a.getOrDefault("plans", "1"));
        int len = weights.config().paddedLength();
        long base = deviceMiB();
        System.out.printf("start: %d MiB%n", base);
        int builds = 0;
        try (TornadoForward forward = new TornadoForward(weights.config(), PackedWeights.pack(weights), cached)) {
            for (int round = 0; round < rounds; round++) {
                for (int rows : sizes) {
                    float[][] x = new float[rows][len];
                    float[] group = new float[rows];
                    boolean[] target = new boolean[rows];
                    for (int r = 0; r < rows; r++) {
                        for (int t = 0; t < len; t++) {
                            x[r][t] = (float) Math.sin(0.01 * t + r);
                        }
                        group[r] = r;
                        target[r] = true;
                    }
                    builds++;
                    try {
                        forward.forward(new ModelBatch(x, group, target));
                    } catch (RuntimeException e) {
                        Throwable c = e;
                        while (c.getCause() != null) {
                            c = c.getCause();
                        }
                        System.out.printf("FAILED at build %d (round %d rows %d): %s%n", builds, round, rows, c);
                        return;
                    }
                    System.out.printf("round %d rows %4d: %6d MiB device, %6d MiB host off-heap%n", round, rows, deviceMiB(),
                            hostOffHeapMiB());
                }
            }
        }
        System.out.printf("closed: %d MiB%n", deviceMiB());
    }

    /** Off-heap memory in use (the JVM's "direct" pool, which counts native segments) after a GC, in MiB. */
    static long hostOffHeapMiB() {
        System.gc();
        for (java.lang.management.BufferPoolMXBean b : java.lang.management.ManagementFactory.getPlatformMXBeans(
                java.lang.management.BufferPoolMXBean.class)) {
            if (b.getName().equals("direct")) {
                return b.getMemoryUsed() >> 20;
            }
        }
        return -1;
    }

    /** This process's device memory in MiB, as nvidia-smi reports it (0 if it has none). */
    static long deviceMiB() throws IOException, InterruptedException {
        if (!new java.io.File("/usr/bin/nvidia-smi").exists()) {
            return 0;
        }
        Process p = new ProcessBuilder("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
                .redirectErrorStream(true).start();
        String out = new String(p.getInputStream().readAllBytes());
        p.waitFor();
        long pid = ProcessHandle.current().pid();
        for (String line : out.split("\n")) {
            String[] f = line.split(",");
            if (f.length == 2 && f[0].trim().equals(String.valueOf(pid))) {
                return Long.parseLong(f[1].trim());
            }
        }
        return 0;
    }
}
