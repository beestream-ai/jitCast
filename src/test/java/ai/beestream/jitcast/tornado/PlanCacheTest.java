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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

/** The plan cache's eviction, budget, failure handling and closing, with stand-in plans. */
public class PlanCacheTest {

    private static final class FakePlan implements PlanCache.Plan {
        final int key;
        final long bytes;
        boolean closed;
        boolean failClose;

        FakePlan(int key, long bytes) {
            this.key = key;
            this.bytes = bytes;
        }

        @Override
        public long deviceBytes() {
            return bytes;
        }

        @Override
        public void close() {
            closed = true;
            if (failClose) {
                throw new IllegalStateException("close " + key);
            }
        }
    }

    private final List<FakePlan> built = new ArrayList<>();

    private FakePlan build(int key, long bytes) {
        FakePlan p = new FakePlan(key, bytes);
        built.add(p);
        return p;
    }

    @Test
    public void reusesAndEvictsLeastRecentlyUsed() {
        PlanCache<Integer, FakePlan> cache = new PlanCache<>(2, Long.MAX_VALUE);
        for (int k : new int[] {1, 2, 1, 3}) {
            cache.run(k, key -> build(key, 10), p -> p.key);
        }
        assertEquals(3, built.size());
        assertTrue("2 was the least recently used", built.get(1).closed);
        assertFalse(built.get(0).closed);
        assertEquals(List.of(1, 3), cache.keys());
    }

    @Test
    public void keepsDeviceMemoryWithinTheBudget() {
        PlanCache<Integer, FakePlan> cache = new PlanCache<>(8, 100);
        cache.run(1, key -> build(key, 40), p -> null);
        cache.run(2, key -> build(key, 40), p -> null);
        cache.run(3, key -> build(key, 40), p -> null);
        assertTrue(built.get(0).closed);
        assertEquals(80, cache.cachedBytes());
        // a plan larger than the budget still runs, alone
        cache.run(4, key -> build(key, 500), p -> null);
        assertEquals(List.of(4), cache.keys());
    }

    @Test
    public void dropsAPlanWhoseFirstRunFailsAndRetriesOnce() {
        PlanCache<Integer, FakePlan> cache = new PlanCache<>(8, Long.MAX_VALUE);
        cache.run(1, key -> build(key, 10), p -> null);
        int[] calls = {0};
        cache.run(2, key -> build(key, 10), p -> {
            if (calls[0]++ == 0) {
                throw new IllegalStateException("out of device memory");
            }
            return null;
        });
        assertTrue("the failed plan is closed", built.get(1).closed);
        assertTrue("the others are evicted for the retry", built.get(0).closed);
        assertEquals(List.of(2), cache.keys());
    }

    @Test
    public void aFailingFirstRunIsNotCached() {
        PlanCache<Integer, FakePlan> cache = new PlanCache<>(8, Long.MAX_VALUE);
        assertThrows(IllegalStateException.class, () -> cache.run(1, key -> build(key, 10), p -> {
            throw new IllegalStateException("bad");
        }));
        assertTrue(built.get(0).closed);
        assertTrue(cache.keys().isEmpty());
    }

    @Test
    public void closeClosesEveryPlanEvenWhenOneFails() {
        PlanCache<Integer, FakePlan> cache = new PlanCache<>(8, Long.MAX_VALUE);
        for (int k = 0; k < 3; k++) {
            cache.run(k, key -> build(key, 10), p -> null);
        }
        built.get(0).failClose = true;
        built.get(1).failClose = true;
        IllegalStateException e = assertThrows(IllegalStateException.class, cache::close);
        assertEquals(1, e.getSuppressed().length);
        for (FakePlan p : built) {
            assertTrue(p.closed);
        }
        assertTrue(cache.keys().isEmpty());
    }
}
