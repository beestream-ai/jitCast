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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitcast.api.TimeSeries;
import org.junit.Test;

/** Splitting a forecast call into forward passes by series count and token budget. */
public class TokenBudgetTest {

    private static final int PATCHES = 74;

    /** A series of {@code targets} targets and {@code covariates} past covariates. */
    private static TimeSeries series(int targets, int covariates) {
        float[][] t = new float[targets][16];
        float[][] c = new float[covariates][16];
        return new TimeSeries(t, c, new float[0][]);
    }

    private static long tokens(TimeSeries ts) {
        return (long) (ts.targets() + ts.pastCount() + ts.futureCount()) * PATCHES;
    }

    @Test
    public void univariateSeriesSplitBySeriesCountOnly() {
        List<TimeSeries> s = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            s.add(series(1, 0));
        }
        List<int[]> w = SeriesWindows.split(s, 512, 131_072, PATCHES);
        assertEquals(2, w.size());
        assertArrayEquals(new int[] {0, 512}, w.get(0));
        assertArrayEquals(new int[] {512, 1000}, w.get(1));
    }

    @Test
    public void covariatesCountTowardsTheBudget() {
        List<TimeSeries> s = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            s.add(series(1, 15)); // 16 rows, 1,184 tokens each
        }
        long budget = 131_072;
        List<int[]> w = SeriesWindows.split(s, 512, budget, PATCHES);
        int next = 0;
        for (int[] win : w) {
            assertEquals("windows are consecutive", next, win[0]);
            long t = 0;
            for (int i = win[0]; i < win[1]; i++) {
                t += tokens(s.get(i));
            }
            assertTrue("window over the budget: " + t, t <= budget);
            next = win[1];
        }
        assertEquals(256, next);
        assertEquals("110 series of 1,184 tokens fit in 131,072", 110, w.get(0)[1]);
    }

    @Test
    public void aSeriesOverTheBudgetRunsAlone() {
        List<TimeSeries> s = List.of(series(1, 0), series(1, 63), series(1, 0));
        List<int[]> w = SeriesWindows.split(s, 512, 64L * PATCHES, PATCHES);
        assertEquals(3, w.size());
        assertArrayEquals(new int[] {1, 2}, w.get(1));
    }
}
