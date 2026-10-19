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
package ai.beestream.jitcast.host;

import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitcast.api.TimeSeries;

/**
 * Splits a forecast call into forward passes. Every model family packs one row per target and
 * covariate and sizes its device buffers by tokens (rows × patches), so a token budget bounds an
 * engine's memory whatever the series' covariate counts.
 */
public final class SeriesWindows {

    private SeriesWindows() {}

    /**
     * The {@code [start, end)} windows a forecast call runs: consecutive series, at most
     * {@code batchSize} of them and at most {@code maxTokens} tokens ({@code patches} per row), but
     * always at least one series, so a series over the budget runs alone.
     */
    public static List<int[]> split(List<TimeSeries> series, int batchSize, long maxTokens, int patches) {
        List<int[]> out = new ArrayList<>();
        int n = series.size();
        for (int start = 0; start < n;) {
            int end = start;
            long tokens = 0;
            while (end < n && end - start < batchSize) {
                TimeSeries ts = series.get(end);
                long t = (long) (ts.targets() + ts.pastCount() + ts.futureCount()) * patches;
                if (end > start && tokens + t > maxTokens) {
                    break;
                }
                tokens += t;
                end++;
            }
            out.add(new int[] {start, end});
            start = end;
        }
        return out;
    }
}
