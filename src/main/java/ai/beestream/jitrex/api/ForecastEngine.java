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
package ai.beestream.jitrex.api;

import java.util.List;

/**
 * A forecasting engine for one model family, such as TiRex-2: its CPU reference or a TornadoVM
 * backend.
 */
public interface ForecastEngine extends AutoCloseable {

    /**
     * Forecasts {@code horizon} steps for every series, with the quantile levels of the model
     * family. For TiRex-2 this is {@code ForecastModel.forecast(series, prediction_length=horizon)}.
     */
    List<Forecast> forecast(List<TimeSeries> series, int horizon);

    /** A short name for logs and benchmark records, e.g. {@code cpu-fp32} or {@code tornado-cuda}. */
    String name();

    @Override
    void close();
}
