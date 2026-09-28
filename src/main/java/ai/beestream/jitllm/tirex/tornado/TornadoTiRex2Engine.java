package ai.beestream.jitllm.tirex.tornado;

import java.util.List;

import ai.beestream.jitllm.tirex.api.Forecast;
import ai.beestream.jitllm.tirex.api.TiRex2Engine;
import ai.beestream.jitllm.tirex.api.TimeSeries;
import ai.beestream.jitllm.tirex.model.TiRex2Weights;

/**
 * The TornadoVM engine (P1). One execution plan per shape bucket (rows × 74 tokens), with the
 * weights resident on the device:
 *
 * <ol>
 *   <li>input stage and patch embedding;
 *   <li>per block: RMSNorm → fused projections (cuBLAS) → mLSTM parallel kernel or sLSTM recurrence
 *       → per-head LayerNorm → combination over known covariates → SwiGLU FFN; then the variate
 *       attention when a group has more than one row;
 *   <li>output norm and embedding, un-patch and rescale.
 * </ol>
 *
 * The host keeps batch assembly and the postprocessor, as the reference does.
 */
public final class TornadoTiRex2Engine implements TiRex2Engine {

    private final TiRex2Weights weights;

    public TornadoTiRex2Engine(TiRex2Weights weights) {
        this.weights = weights;
    }

    @Override
    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        throw new UnsupportedOperationException("the TornadoVM forward pass is not implemented yet (P1)");
    }

    @Override
    public String name() {
        return "tornado";
    }

    @Override
    public void close() {
        weights.close();
    }
}
