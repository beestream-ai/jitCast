package ai.beestream.jitllm.tirex.api;

/**
 * One forecasting input, as {@code tirex2.TimeseriesType}: target rows plus optional covariates.
 * Missing observations are {@code NaN}.
 *
 * @param target {@code [V_t][T]}; a univariate series has one row
 * @param pastCovariates {@code [V_p][T]}, known only up to the forecast start, or null
 * @param futureCovariates {@code [V_f][>= T + H]}, known over the horizon too, or null; extra
 *     trailing steps are ignored
 */
public record TimeSeries(float[][] target, float[][] pastCovariates, float[][] futureCovariates) {

    public TimeSeries {
        if (target == null || target.length == 0) {
            throw new IllegalArgumentException("a series needs at least one target row");
        }
        int t = target[0].length;
        for (float[] row : target) {
            requireLength(row, t, "target");
        }
        if (pastCovariates != null) {
            for (float[] row : pastCovariates) {
                requireLength(row, t, "past covariate");
            }
        }
        if (futureCovariates != null) {
            for (float[] row : futureCovariates) {
                if (row.length < t) {
                    throw new IllegalArgumentException("a future covariate row is shorter than the target");
                }
            }
        }
    }

    public static TimeSeries univariate(float[] values) {
        return new TimeSeries(new float[][] {values}, null, null);
    }

    public int length() {
        return target[0].length;
    }

    public int targets() {
        return target.length;
    }

    public int pastCount() {
        return pastCovariates == null ? 0 : pastCovariates.length;
    }

    public int futureCount() {
        return futureCovariates == null ? 0 : futureCovariates.length;
    }

    private static void requireLength(float[] row, int t, String what) {
        if (row.length != t) {
            throw new IllegalArgumentException("every " + what + " row must have the target's length " + t);
        }
    }
}
