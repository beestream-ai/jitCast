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

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * A family's compiled plans, one per input shape, least recently used evicted first, behind one
 * lock: a plan's buffers are reused by every call of its shape, so calls on one engine run one at a
 * time, and an engine can be shared between threads.
 *
 * <p>The cache holds at most {@code maxPlans} plans and keeps their device memory within
 * {@code maxBytes}. TornadoVM gives every execution plan its own device buffers, the model's weights
 * included, so each plan counts its whole footprint, as TornadoVM reports it after the plan's first
 * run. A plan whose first run fails is closed and dropped rather than cached; when other plans were
 * cached, they are evicted and the plan is built and run once more, in case the device was out of
 * memory.
 *
 * @param <K> the shape
 * @param <P> the plan
 */
public final class PlanCache<K, P extends PlanCache.Plan> implements AutoCloseable {

    /** A compiled plan: its device buffers are released on {@link #close()}. */
    public interface Plan extends AutoCloseable {
        /** The device memory the plan holds, its weights included. */
        long deviceBytes();

        @Override
        void close();
    }

    private record Entry<P>(P plan, long bytes) {}

    private final int maxPlans;
    private final long maxBytes;
    private final Map<K, Entry<P>> plans = new LinkedHashMap<>(16, 0.75f, true);
    private long cachedBytes;

    public PlanCache(int maxPlans, long maxBytes) {
        if (maxPlans < 1 || maxBytes < 1) {
            throw new IllegalArgumentException("a plan cache needs room for one plan");
        }
        this.maxPlans = maxPlans;
        this.maxBytes = maxBytes;
    }

    /**
     * Runs {@code use} on the plan of {@code key}, built with {@code build} on its first use. The
     * whole call holds the cache's lock.
     */
    public synchronized <R> R run(K key, Function<K, P> build, Function<P, R> use) {
        Entry<P> e = plans.get(key);
        if (e != null) {
            return use.apply(e.plan());
        }
        evictWhile(() -> plans.size() >= maxPlans);
        P plan = build.apply(key);
        R result;
        try {
            result = use.apply(plan);
        } catch (RuntimeException | Error first) {
            closeAfter(plan, first);
            if (plans.isEmpty()) {
                throw first;
            }
            clear(first);
            plan = build.apply(key);
            try {
                result = use.apply(plan);
            } catch (RuntimeException | Error second) {
                second.addSuppressed(first);
                closeAfter(plan, second);
                throw second;
            }
        }
        long bytes = plan.deviceBytes();
        evictWhile(() -> !plans.isEmpty() && cachedBytes + bytes > maxBytes);
        plans.put(key, new Entry<>(plan, bytes));
        cachedBytes += bytes;
        return result;
    }

    /** The shapes cached, least recently used first. */
    public synchronized List<K> keys() {
        return List.copyOf(plans.keySet());
    }

    /** The device memory the cached plans hold. */
    public synchronized long cachedBytes() {
        return cachedBytes;
    }

    private void evictWhile(java.util.function.BooleanSupplier full) {
        Iterator<Entry<P>> it = plans.values().iterator();
        while (full.getAsBoolean() && it.hasNext()) {
            Entry<P> eldest = it.next();
            it.remove();
            cachedBytes -= eldest.bytes();
            eldest.plan().close();
        }
    }

    private static void closeAfter(Plan plan, Throwable failure) {
        try {
            plan.close();
        } catch (RuntimeException | Error e) {
            failure.addSuppressed(e);
        }
    }

    /** Closes every plan, attaching any further failures to {@code failure}. */
    private void clear(Throwable failure) {
        List<Entry<P>> all = new ArrayList<>(plans.values());
        plans.clear();
        cachedBytes = 0;
        for (Entry<P> e : all) {
            closeAfter(e.plan(), failure);
        }
    }

    /** Closes every plan, even when some fail to close; the first failure is thrown, the others suppressed in it. */
    @Override
    public synchronized void close() {
        List<Entry<P>> all = new ArrayList<>(plans.values());
        plans.clear();
        cachedBytes = 0;
        RuntimeException failure = null;
        for (Entry<P> e : all) {
            try {
                e.plan().close();
            } catch (RuntimeException ex) {
                if (failure == null) {
                    failure = ex;
                } else {
                    failure.addSuppressed(ex);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }
}
