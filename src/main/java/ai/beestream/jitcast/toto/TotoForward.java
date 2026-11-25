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
package ai.beestream.jitcast.toto;

/**
 * Toto 2.0's network, from scaled patches to the output head's quantiles on the forecast patches. The
 * causal scaler and everything in real space stay on the host ({@link TotoPipeline}).
 */
public interface TotoForward extends AutoCloseable {

    /** The {@code 1/√2} that a residual MLP's two branches are summed with. */
    double RESIDUAL_MLP_SCALE = Math.sqrt(0.5);

    /**
     * The forward scale of a unit-scaled SiLU at {@code mult = 1}: the output standard deviation model of
     * {@code unit_scaling.functional.silu}, interpolated logarithmically between {@code 2} and
     * {@code √(2 / (1 − 1/π))} at {@code α = 0.8}.
     */
    double SILU_SCALE = Math.pow(2.0, 0.2) * Math.pow(Math.sqrt(2.0 / (1.0 - 1.0 / Math.PI)), 0.8);

    /** The attention logits' scale: {@code 1/head_dim} (u-μP), not {@code 1/√head_dim}. */
    static double attentionScale(TotoConfig c) {
        return 1.0 / c.headDim();
    }

    /**
     * One batch: {@code rows} variate rows of {@code patches} patches each, left-aligned to a common length,
     * whose last {@code forecast} patches are the forecast region. A row's patch is valid when it holds at
     * least one observed step (the forecast patches always are); time attention connects valid patches
     * with valid ones and invalid with invalid, causally, and variate attention connects the rows of one
     * series the same way, patch by patch.
     *
     * @param rows the variate rows
     * @param patches the patches per row, forecast patches included
     * @param forecast the forecast patches at the end of each row
     * @param values {@code [rows, patches, patch]}: the scaled, asinh-transformed values, 0 where unobserved
     * @param missing {@code [rows, patches, patch]}: 1 where unobserved, else 0
     * @param valid {@code [rows, patches]}: whether each patch is valid
     * @param seriesStart {@code [series + 1]}: the first row of each series; a series' rows are contiguous
     */
    record Input(int rows, int patches, int forecast, float[] values, float[] missing, boolean[] valid, int[] seriesStart) {

        /**
         * Validates the shapes.
         */
        public Input {
            if (rows < 1 || patches < 1 || forecast < 1 || forecast >= patches) {
                throw new IllegalArgumentException("need rows ≥ 1 and 1 ≤ forecast < patches");
            }
            if (valid.length != rows * patches || values.length != missing.length || values.length % (rows * patches) != 0) {
                throw new IllegalArgumentException("inconsistent input sizes");
            }
            if (seriesStart[0] != 0 || seriesStart[seriesStart.length - 1] != rows) {
                throw new IllegalArgumentException("seriesStart must run from 0 to rows");
            }
        }

        /** {@return the number of series} */
        public int series() {
            return seriesStart.length - 1;
        }

        /**
         * {@return the series that row {@code r} belongs to}
         *
         * @param r the row
         */
        public int seriesOf(int r) {
            int s = 0;
            while (seriesStart[s + 1] <= r) {
                s++;
            }
            return s;
        }
    }

    /**
     * Runs the network.
     *
     * @param in the batch
     * @return {@code [rows, forecast, patch, quantiles]}: the output head on the patches that predict the
     *         forecast patches (patches {@code patches − forecast − 1} to {@code patches − 2}), in the scaled,
     *         asinh-transformed space and not yet sorted
     */
    float[] forward(Input in);

    @Override
    default void close() {}
}
