package ai.beestream.jitrex.cpu;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import ai.beestream.jitrex.api.Forecast;
import ai.beestream.jitrex.api.TiRex2Engine;
import ai.beestream.jitrex.api.TimeSeries;
import ai.beestream.jitrex.host.ForecastPipeline;
import ai.beestream.jitrex.model.TiRex2Weights;

/** The FP32 CPU reference as a complete forecasting engine. */
public final class CpuTiRex2Engine implements TiRex2Engine {

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
