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
package ai.beestream.jitcast.timestral.tornado;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.ForecastEngine;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.timestral.TimestralConfig;
import ai.beestream.jitcast.timestral.TimestralPipeline;
import ai.beestream.jitcast.timestral.TimestralWeights;

/** The TornadoVM Timestral engine: the shared host pipeline ({@link TimestralPipeline}) around {@link TimestralForward}. */
public final class TornadoTimestralEngine implements ForecastEngine {

    private final TimestralWeights weights;
    private final TimestralForward forward;
    private final TimestralPipeline pipeline;
    private boolean closed;

    public TornadoTimestralEngine(TimestralWeights weights) {
        this.weights = weights;
        this.forward = new TimestralForward(weights.config(), TimestralPacked.pack(weights));
        this.pipeline = new TimestralPipeline(weights.config(), forward);
    }

    /** Loads a checkpoint directory holding {@code config.json} and {@code model.safetensors}. */
    public static TornadoTimestralEngine load(Path checkpoint) throws IOException {
        TimestralWeights weights = TimestralWeights.load(checkpoint);
        try {
            return new TornadoTimestralEngine(weights);
        } catch (RuntimeException | Error e) {
            try {
                weights.close();
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    public TimestralConfig config() {
        return weights.config();
    }

    public TimestralForward forwardPass() {
        return forward;
    }

    public TimestralPipeline pipeline() {
        return pipeline;
    }

    @Override
    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        return pipeline.forecast(series, horizon);
    }

    @Override
    public String name() {
        return "timestral-tornado";
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            forward.close();
        } finally {
            weights.close();
        }
    }
}
