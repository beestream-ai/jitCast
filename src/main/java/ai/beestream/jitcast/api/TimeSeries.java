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
package ai.beestream.jitcast.api;

/**
 * One forecasting input: target rows plus optional numeric and categorical covariates. Missing
 * observations are {@code NaN}.
 *
 * @param target {@code [V_t][T]}; a univariate series has one row
 * @param pastCovariates {@code [V_p][T]}, known only up to the forecast start, or null
 * @param futureCovariates {@code [V_f][>= T + H]}, known over the horizon too, or null; extra
 *     trailing steps are ignored
 * @param pastCategorical {@code [C_p][T]} category codes (any numbers; {@code NaN} is missing),
 *     known only up to the forecast start, or null
 * @param futureCategorical {@code [C_f][>= T + H]} category codes known over the horizon too, or null
 */
public record TimeSeries(float[][] target, float[][] pastCovariates, float[][] futureCovariates, double[][] pastCategorical,
        double[][] futureCategorical) {

    /**
     * Checks the shapes.
     *
     * @throws IllegalArgumentException if there is no target row or time step, a row has the wrong
     *     length, or a category code is infinite
     */
    public TimeSeries {
        if (target == null || target.length == 0) {
            throw new IllegalArgumentException("a series needs at least one target row");
        }
        int t = target[0].length;
        if (t == 0) {
            throw new IllegalArgumentException("a series needs at least one time step");
        }
        for (float[] row : target) {
            requireLength(row.length, t, "target");
        }
        if (pastCovariates != null) {
            for (float[] row : pastCovariates) {
                requireLength(row.length, t, "past covariate");
            }
        }
        if (futureCovariates != null) {
            for (float[] row : futureCovariates) {
                if (row.length < t) {
                    throw new IllegalArgumentException("a future covariate row is shorter than the target");
                }
            }
        }
        if (pastCategorical != null) {
            for (double[] row : pastCategorical) {
                requireLength(row.length, t, "past categorical covariate");
                requireNotInfinite(row);
            }
        }
        if (futureCategorical != null) {
            for (double[] row : futureCategorical) {
                if (row.length < t) {
                    throw new IllegalArgumentException("a future categorical covariate row is shorter than the target");
                }
                requireNotInfinite(row);
            }
        }
    }

    /**
     * A series with numeric covariates only.
     *
     * @param target {@code [V_t][T]}
     * @param pastCovariates {@code [V_p][T]}, or null
     * @param futureCovariates {@code [V_f][>= T + H]}, or null
     */
    public TimeSeries(float[][] target, float[][] pastCovariates, float[][] futureCovariates) {
        this(target, pastCovariates, futureCovariates, null, null);
    }

    /**
     * {@return a series with one target and no covariates}
     *
     * @param values the history, {@code NaN} where missing
     */
    public static TimeSeries univariate(float[] values) {
        return new TimeSeries(new float[][] {values}, null, null);
    }

    /** {@return the number of observed time steps, {@code T}} */
    public int length() {
        return target[0].length;
    }

    /** {@return the number of target rows} */
    public int targets() {
        return target.length;
    }

    /** {@return the number of numeric past-only covariates} */
    public int pastCount() {
        return pastCovariates == null ? 0 : pastCovariates.length;
    }

    /** {@return the number of numeric known-future covariates} */
    public int futureCount() {
        return futureCovariates == null ? 0 : futureCovariates.length;
    }

    /** {@return the number of categorical past-only covariates} */
    public int pastCategoricalCount() {
        return pastCategorical == null ? 0 : pastCategorical.length;
    }

    /** {@return the number of categorical known-future covariates} */
    public int futureCategoricalCount() {
        return futureCategorical == null ? 0 : futureCategorical.length;
    }

    /** {@return whether the series has categorical covariates} */
    public boolean hasCategorical() {
        return pastCategoricalCount() + futureCategoricalCount() > 0;
    }

    /** {@return the rows the series contributes to a forward pass: targets and all covariates} */
    public int rows() {
        return targets() + pastCount() + futureCount() + pastCategoricalCount() + futureCategoricalCount();
    }

    private static void requireLength(int length, int t, String what) {
        if (length != t) {
            throw new IllegalArgumentException("every " + what + " row must have the target's length " + t);
        }
    }

    private static void requireNotInfinite(double[] codes) {
        for (double c : codes) {
            if (Double.isInfinite(c)) {
                throw new IllegalArgumentException("category codes must be finite or NaN");
            }
        }
    }
}
