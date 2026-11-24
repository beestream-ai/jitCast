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
package ai.beestream.jitcast.t0.cpu;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.ForecastEngine;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.t0.T0Config;
import ai.beestream.jitcast.t0.T0Pipeline;
import ai.beestream.jitcast.t0.T0Weights;

/** The FP32 CPU reference of t0 as a complete forecasting engine. */
public final class CpuT0Engine implements ForecastEngine {

    private final T0Weights weights;
    private boolean closed;
    private final T0Pipeline pipeline;

    /**
     * An engine over weights already loaded. The engine owns the weights from then on and closes
     * them in {@link #close()}.
     *
     * @param weights the t0 weights
     */
    public CpuT0Engine(T0Weights weights) {
        this.weights = weights;
        this.pipeline = new T0Pipeline(weights.config(), new CpuT0(weights));
    }

    /**
     * Loads a checkpoint directory holding {@code model.safetensors} and {@code config.json}, as published, and builds the engine over it.
     *
     * @param checkpoint the checkpoint directory
     * @return the engine; close it when done
     * @throws IOException if the files cannot be read
     * @throws IllegalArgumentException if the checkpoint has settings or tensors the engine does not
     *     support
     */
    public static CpuT0Engine load(Path checkpoint) throws IOException {
        T0Weights weights = T0Weights.load(checkpoint);
        try {
            return new CpuT0Engine(weights);
        } catch (RuntimeException | Error e) {
            try {
                weights.close();
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    /** {@return the checkpoint's configuration} */
    public T0Config config() {
        return weights.config();
    }

    /**
     * {@return the host pipeline {@link #forecast} runs} Its {@code with…} methods return a
     * pipeline with other settings, over this engine's forward pass, to forecast through directly.
     */
    public T0Pipeline pipeline() {
        return pipeline;
    }

    @Override
    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        return pipeline.forecast(series, horizon);
    }

    @Override
    public String name() {
        return "t0-cpu-fp32";
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        weights.close();
    }
}
