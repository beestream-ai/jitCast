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
package ai.beestream.jitcast.toto.cpu;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.ForecastEngine;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.toto.TotoConfig;
import ai.beestream.jitcast.toto.TotoPipeline;
import ai.beestream.jitcast.toto.TotoWeights;

/**
 * Toto 2.0 on the CPU: the reference engine the accelerated one is checked against.
 */
public final class CpuTotoEngine implements ForecastEngine {

    private final TotoWeights weights;
    private final TotoPipeline pipeline;
    private boolean closed;

    /**
     * An engine over loaded weights, which it closes.
     *
     * @param weights the checkpoint
     */
    public CpuTotoEngine(TotoWeights weights) {
        this.weights = weights;
        this.pipeline = new TotoPipeline(weights.config(), new CpuToto(weights));
    }

    /**
     * {@return an engine over a checkpoint directory}
     *
     * @param checkpoint a directory holding {@code config.json} and {@code model.safetensors}
     * @throws IOException if it cannot be read
     */
    public static CpuTotoEngine load(Path checkpoint) throws IOException {
        TotoWeights weights = TotoWeights.load(checkpoint);
        try {
            return new CpuTotoEngine(weights);
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
    public TotoConfig config() {
        return weights.config();
    }

    /** {@return the pipeline} */
    public TotoPipeline pipeline() {
        return pipeline;
    }

    @Override
    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        return pipeline.forecast(series, horizon);
    }

    @Override
    public String name() {
        return "toto-cpu-fp32";
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
