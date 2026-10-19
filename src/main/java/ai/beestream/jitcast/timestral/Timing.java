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
package ai.beestream.jitcast.timestral;

import java.util.Map;
import java.util.TreeMap;

/**
 * Wall time per phase of a forecast ({@code -Dtimestral.timing=true}, analysis only): the engines
 * add each phase's time, and {@link #report} prints and clears the totals.
 */
public final class Timing {

    public static final boolean ENABLED = Boolean.getBoolean("timestral.timing");

    private static final Map<String, long[]> TOTALS = new TreeMap<>();

    private Timing() {}

    /** The current time, or 0 when timing is off. */
    public static long start() {
        return ENABLED ? System.nanoTime() : 0;
    }

    /** Adds the time since {@code start} to {@code phase}. */
    public static void add(String phase, long start) {
        if (ENABLED) {
            long dt = System.nanoTime() - start;
            synchronized (TOTALS) {
                long[] t = TOTALS.computeIfAbsent(phase, k -> new long[2]);
                t[0] += dt;
                t[1]++;
            }
        }
    }

    /** Prints each phase's total ms per call of {@code calls} and clears the totals. */
    public static void report(String label, int calls) {
        if (!ENABLED) {
            return;
        }
        StringBuilder sb = new StringBuilder("[timing] " + label + " ms per forecast:");
        synchronized (TOTALS) {
            for (Map.Entry<String, long[]> e : TOTALS.entrySet()) {
                sb.append(String.format(" %s=%.2f", e.getKey(), e.getValue()[0] / 1e6 / calls));
            }
            TOTALS.clear();
        }
        System.err.println(sb);
    }
}
