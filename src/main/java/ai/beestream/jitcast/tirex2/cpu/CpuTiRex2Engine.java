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
package ai.beestream.jitcast.tirex2.cpu;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.ForecastEngine;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.tirex2.ForecastPipeline;
import ai.beestream.jitcast.tirex2.TiRex2Weights;

/** The FP32 CPU reference as a complete forecasting engine. */
public final class CpuTiRex2Engine implements ForecastEngine {

    private final TiRex2Weights weights;
    private final ForecastPipeline pipeline;

    public CpuTiRex2Engine(TiRex2Weights weights) {
        this.weights = weights;
        this.pipeline = new ForecastPipeline(weights.config(), new CpuReference(weights));
    }

    public static CpuTiRex2Engine load(Path safetensors) throws IOException {
        return new CpuTiRex2Engine(TiRex2Weights.load(safetensors));
    }

    public ForecastPipeline pipeline() {
        return pipeline;
    }

    @Override
    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        return pipeline.forecast(series, horizon);
    }

    @Override
    public String name() {
        return "cpu-fp32";
    }

    @Override
    public void close() {
        weights.close();
    }
}
