package ai.beestream.jitllm.tirex.api;

import java.util.List;

/** A TiRex-2 forecasting engine: the CPU reference or a TornadoVM backend. */
public interface TiRex2Engine extends AutoCloseable {

    /**
     * Forecasts {@code horizon} steps for every series in one batched forward pass, as
     * {@code ForecastModel.forecast(series, prediction_length=horizon)}.
     */
    List<Forecast> forecast(List<TimeSeries> series, int horizon);

    /** A short name for logs and benchmark records, e.g. {@code cpu-fp32} or {@code tornado-cuda}. */
    String name();

    @Override
    void close();
}
