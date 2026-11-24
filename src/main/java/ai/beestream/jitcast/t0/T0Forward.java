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
package ai.beestream.jitcast.t0;

import java.util.Arrays;

/**
 * One engine's t0 forward pass ({@code T0Forecaster.forward}) over a scaled window: the quantiles each patch
 * predicts for the patch after it.
 */
public interface T0Forward extends AutoCloseable {

    /**
     * The inputs of one forward pass, patched, with what the attention masks need.
     *
     * @param rows the number of rows (variates of every series)
     * @param patches the patches per row
     * @param values {@code [rows · patches · patchSize]} the scaled values
     * @param valid {@code [rows · patches · patchSize]} 1 where a step is an observation, else 0
     * @param patchType {@code [rows · patches]} each patch's variate type, 0 for an all-padding patch (the type
     *     embedding it gets)
     * @param patchValid {@code [rows · patches]} whether a patch holds anything but padding: only those attend
     *     and are attended to
     * @param future {@code [rows]} whether a row is a known-future covariate: its time attention is
     *     bidirectional, and across variates it reads only other known-future covariates
     * @param groupOf {@code [rows]} the dense index of each row's series
     * @param groupStart {@code [groups + 1]} where each series' rows start in {@code groupRows}
     * @param groupRows {@code [rows]} the rows, ordered by series
     */
    record Input(int rows, int patches, float[] values, float[] valid, int[] patchType, boolean[] patchValid,
            boolean[] future, int[] groupOf, int[] groupStart, int[] groupRows) {

        /**
         * {@return the forward pass's inputs for a scaled window}
         *
         * @param b the window, aligned to whole patches
         * @param scaled its scaled values
         * @param p the patch size
         */
        public static Input of(T0Batch b, float[] scaled, int p) {
            int rows = b.rows(), w = b.width(), n = w / p;
            if (w % p != 0) {
                throw new IllegalArgumentException("the window is not a whole number of patches");
            }
            float[] valid = new float[rows * w];
            int[] patchType = new int[rows * n];
            boolean[] patchValid = new boolean[rows * n], future = new boolean[rows];
            for (int i = 0; i < valid.length; i++) {
                valid[i] = b.mask()[i] == T0Batch.VALID ? 1f : 0f;
            }
            for (int r = 0; r < rows; r++) {
                for (int s = 0; s < n; s++) {
                    int first = -1;
                    for (int j = 0; j < p && first < 0; j++) {
                        if (b.mask()[r * w + s * p + j] != T0Batch.PAD) {
                            first = r * w + s * p + j;
                        }
                    }
                    // reduce_patch_metadata: the patch's first non-padding step, or -1 when all padding.
                    int group = first < 0 ? -1 : b.group()[first], type = first < 0 ? -1 : b.type()[first];
                    patchValid[r * n + s] = group >= 0;
                    patchType[r * n + s] = Math.max(type, 0);
                }
                future[r] = b.rowType(r) == T0Batch.FUTURE;
            }
            int[] groupOf = new int[rows];
            int[] ids = new int[rows];
            int groups = 0;
            for (int r = 0; r < rows; r++) {
                int g = b.rowGroup(r), k = 0;
                while (k < groups && ids[k] != g) {
                    k++;
                }
                if (k == groups) {
                    ids[groups++] = g;
                }
                groupOf[r] = k;
            }
            int[] groupStart = new int[groups + 1], groupRows = new int[rows];
            for (int r = 0; r < rows; r++) {
                groupStart[groupOf[r] + 1]++;
            }
            for (int g = 0; g < groups; g++) {
                groupStart[g + 1] += groupStart[g];
            }
            int[] fill = Arrays.copyOf(groupStart, groups);
            for (int r = 0; r < rows; r++) {
                groupRows[fill[groupOf[r]]++] = r;
            }
            return new Input(rows, n, scaled, valid, patchType, patchValid, future, groupOf, groupStart, groupRows);
        }

        /** {@return the number of series} */
        public int groups() {
            return groupStart.length - 1;
        }
    }

    /**
     * {@return the quantile head's output, still scaled, for {@code count} patches from {@code from}}:
     * {@code [rows · count · patchSize · quantiles]}
     *
     * @param in the inputs
     * @param from the first patch whose prediction is wanted
     * @param count the number of patches
     */
    float[] forward(Input in, int from, int count);

    @Override
    default void close() {}
}
