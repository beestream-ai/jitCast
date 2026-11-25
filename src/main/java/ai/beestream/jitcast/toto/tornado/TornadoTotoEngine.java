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
package ai.beestream.jitcast.toto.tornado;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.ForecastEngine;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.toto.TotoConfig;
import ai.beestream.jitcast.toto.TotoPipeline;
import ai.beestream.jitcast.toto.TotoWeights;

/** The TornadoVM Toto 2.0 engine: the shared host pipeline ({@link TotoPipeline}) around {@link TornadoTotoForward}. */
public final class TornadoTotoEngine implements ForecastEngine {

    private final TotoWeights weights;
    private final TornadoTotoForward forward;
    private final TotoPipeline pipeline;
    private boolean closed;

    /**
     * An engine over weights already loaded. The engine owns the weights from then on and closes
     * them in {@link #close()}.
     *
     * @param weights the Toto 2.0 weights
     */
    public TornadoTotoEngine(TotoWeights weights) {
        this.weights = weights;
        this.forward = new TornadoTotoForward(weights.config(), TotoPacked.pack(weights));
        this.pipeline = new TotoPipeline(weights.config(), forward);
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
    public static TornadoTotoEngine load(Path checkpoint) throws IOException {
        TotoWeights weights = TotoWeights.load(checkpoint);
        try {
            return new TornadoTotoEngine(weights);
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

    /** {@return the device forward pass under the pipeline, for benchmarks and tests} */
    public TornadoTotoForward forwardPass() {
        return forward;
    }

    /**
     * {@return the host pipeline {@link #forecast} runs} Its {@code with…} methods return a
     * pipeline with other settings, over this engine's forward pass, to forecast through directly.
     */
    public TotoPipeline pipeline() {
        return pipeline;
    }

    @Override
    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        return pipeline.forecast(series, horizon);
    }

    @Override
    public String name() {
        return "toto-tornado";
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
