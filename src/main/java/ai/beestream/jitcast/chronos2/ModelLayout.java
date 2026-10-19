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
package ai.beestream.jitcast.chronos2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The token layout and attention masks of one forward pass, shared by every engine so they cannot
 * disagree.
 *
 * <p>The context is cut to the model's {@code context_length} (the most recent steps) and padded
 * with {@code NaN} on the left to whole patches. Each row then has {@link #contextPatches()}
 * context tokens, the {@code [REG]} token when the checkpoint has one, and {@link #outputPatches()}
 * future tokens; a token's position is its index.
 *
 * <ul>
 *   <li>A token is a valid key ({@link #live}) unless it is a context patch with no observed value.
 *   <li>Time attention, on every query of a row: the valid keys of the row, in both directions.
 *   <li>Group attention, per token slot: the rows of the same group whose token at that slot is a
 *       valid key.
 * </ul>
 *
 * A token that is not a valid key is never read by another token, so its value does not reach the
 * forecasts. When none of its group's tokens at its slot is valid either, the reference's group
 * attention averages every row of the batch for it (its additive masks swamp the scores); the
 * engines write zeros there instead, so those tokens are left out when activations are compared.
 */
public final class ModelLayout {

    private final int rows;
    private final int patch;
    private final int contextPatches;
    private final int reg;
    private final int outputPatches;
    private final int[] group;
    private final boolean[][] live;
    private final float[][] window;
    private final float[][] future;

    private ModelLayout(int rows, int patch, int contextPatches, int reg, int outputPatches, int[] group, boolean[][] live,
            float[][] window, float[][] future) {
        this.rows = rows;
        this.patch = patch;
        this.contextPatches = contextPatches;
        this.reg = reg;
        this.outputPatches = outputPatches;
        this.group = group;
        this.live = live;
        this.window = window;
        this.future = future;
    }

    public static ModelLayout of(Chronos2Config config, ModelInput in) {
        int ps = config.patchSize();
        int r = in.rows();
        int h = in.outputPatches();
        if (h < 1) {
            throw new IllegalArgumentException("output patches must be >= 1, got " + h);
        }
        if (in.group().length != r) {
            throw new IllegalArgumentException("group ids must have one entry per row");
        }
        int t = in.length();
        int kept = Math.min(t, config.contextLength());
        int padded = (kept + ps - 1) / ps * ps;
        int nc = padded / ps;
        float[][] window = new float[r][padded];
        for (int i = 0; i < r; i++) {
            if (in.context()[i].length != t) {
                throw new IllegalArgumentException("every context row must have length " + t);
            }
            Arrays.fill(window[i], 0, padded - kept, Float.NaN);
            System.arraycopy(in.context()[i], t - kept, window[i], padded - kept, kept);
        }
        float[][] future = new float[r][h * ps];
        for (int i = 0; i < r; i++) {
            Arrays.fill(future[i], Float.NaN);
            if (in.futureCovariates() != null) {
                float[] f = in.futureCovariates()[i];
                if (f.length > h * ps) {
                    throw new IllegalArgumentException(h + " output patches cannot hold " + f.length + " future steps");
                }
                System.arraycopy(f, 0, future[i], 0, f.length);
            }
        }
        int reg = config.useRegToken() ? 1 : 0;
        boolean[][] live = new boolean[r][nc + reg + h];
        for (int i = 0; i < r; i++) {
            for (int p = 0; p < nc; p++) {
                boolean any = false;
                for (int s = p * ps; s < (p + 1) * ps; s++) {
                    any |= !Float.isNaN(window[i][s]);
                }
                live[i][p] = any;
            }
            Arrays.fill(live[i], nc, nc + reg + h, true);
        }
        return new ModelLayout(r, ps, nc, reg, h, in.group().clone(), live, window, future);
    }

    public int rows() {
        return rows;
    }

    public int patch() {
        return patch;
    }

    /** The context tokens per row. */
    public int contextPatches() {
        return contextPatches;
    }

    /** 1 with a {@code [REG]} token after the context tokens, else 0. */
    public int regTokens() {
        return reg;
    }

    public int outputPatches() {
        return outputPatches;
    }

    /** Context, {@code [REG]} and future tokens per row. */
    public int tokens() {
        return contextPatches + reg + outputPatches;
    }

    /** The first future token of a row. */
    public int futureStart() {
        return contextPatches + reg;
    }

    /** {@code [rows][contextPatches · P]}: the cut and padded context, {@code NaN} where missing. */
    public float[][] window() {
        return window;
    }

    /** {@code [rows][outputPatches · P]}: the known future values, {@code NaN} where unknown. */
    public float[][] future() {
        return future;
    }

    public int group(int row) {
        return group[row];
    }

    /** Whether token {@code token} of row {@code row} is a valid key. */
    public boolean live(int row, int token) {
        return live[row][token];
    }

    /** For each row, the rows of its group in ascending order. */
    public int[][] members() {
        Map<Integer, List<Integer>> byGroup = new HashMap<>();
        for (int i = 0; i < rows; i++) {
            byGroup.computeIfAbsent(group[i], g -> new ArrayList<>()).add(i);
        }
        int[][] m = new int[rows][];
        for (int i = 0; i < rows; i++) {
            m[i] = byGroup.get(group[i]).stream().mapToInt(Integer::intValue).toArray();
        }
        return m;
    }
}
