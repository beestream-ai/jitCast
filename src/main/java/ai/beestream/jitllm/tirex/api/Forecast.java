package ai.beestream.jitllm.tirex.api;

/**
 * Quantile forecasts for one {@link TimeSeries}: {@code values[target][quantile][step]}, with
 * the quantile levels in {@link #levels()} (0.1 … 0.9 for TiRex-2).
 */
public record Forecast(float[] levels, float[][][] values) {

    public int targets() {
        return values.length;
    }

    public int horizon() {
        return values[0][0].length;
    }

    /** The median path of target {@code t} (the 0.5 level). */
    public float[] median(int t) {
        for (int q = 0; q < levels.length; q++) {
            if (Math.abs(levels[q] - 0.5f) < 1e-6f) {
                return values[t][q];
            }
        }
        throw new IllegalStateException("no 0.5 quantile level");
    }
}
