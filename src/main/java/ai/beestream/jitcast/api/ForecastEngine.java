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
package ai.beestream.jitcast.api;

import java.util.List;

/**
 * A forecasting engine for one model family: its CPU reference or its TornadoVM engine.
 *
 * <p>Every engine of every family implements this interface, so the same code forecasts with any
 * model:
 *
 * {@snippet :
 * try (ForecastEngine engine = TornadoTimestralEngine.load(Path.of("Timestral-5M"))) {
 *     List<Forecast> forecasts = engine.forecast(List.of(TimeSeries.univariate(history)), 64);
 *     float[] median = forecasts.get(0).median(0);
 * }
 * }
 *
 * <p>An engine can be shared between threads; a TornadoVM engine runs their calls one at a time,
 * since they reuse the same device buffers. Close an engine to release its device memory and
 * mapped weights; {@link #close()} may be called more than once.
 */
public interface ForecastEngine extends AutoCloseable {

    /**
     * Forecasts every series, with the quantile levels of the model family. A series' forecast does
     * not depend on the rest of the list, except with Chronos-2's cross-learning.
     *
     * @param series the inputs, each with its targets and optional covariates
     * @param horizon the number of steps to forecast, at least 1; TiRex-2 cuts it to its
     *     checkpoint's future length
     * @return one forecast per series, in order
     * @throws IllegalArgumentException if {@code horizon} is below 1, or a series has inputs the
     *     model does not take (such as categorical covariates for TiRex-2 or Chronos-2)
     */
    List<Forecast> forecast(List<TimeSeries> series, int horizon);

    /** {@return a short name for logs and benchmark records, such as {@code cpu-fp32} or {@code tornado}} */
    String name();

    /** Releases the engine's device memory and weights. Later calls do nothing. */
    @Override
    void close();
}
