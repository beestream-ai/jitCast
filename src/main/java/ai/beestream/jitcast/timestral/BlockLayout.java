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
package ai.beestream.jitcast.timestral;

import java.util.HashMap;
import java.util.Map;

/**
 * The token layout and attention masks of one decode block (Decoupled Horizon Forecasting), shared
 * by every engine so they cannot disagree.
 *
 * <p>Each row has {@link #encoderTokens()} encoder tokens, one per patch of its window, followed by
 * {@link #horizonPatches()} decoder tokens, the forecast queries of the horizon patches. Without
 * covariates the window is the context ({@code ll} patches). With covariates it is the context plus
 * the block's horizon, so known-future values are encoder keys; the reference then also runs a
 * decoder token for every context patch, but those tokens attend only to themselves and nothing
 * reads them, so the engines leave them out.
 *
 * <ul>
 *   <li>Time attention, encoder query {@code i}: the valid encoder keys {@code j ≤ i} (every valid
 *       key on a known-covariate row, which sees its whole window), and always itself.
 *   <li>Time attention, decoder query {@code i}: the valid encoder keys of the context ({@code j <
 *       ll}) and the decoder tokens {@code ≤ i}.
 *   <li>Variate attention, per token slot: the rows of the same series, an encoder slot only where
 *       the key row's patch is a valid key; a covariate row never reads a target row, and a
 *       known-covariate row never reads a row that is not one. A row always reads itself.
 * </ul>
 *
 * A patch is a valid key once its row has started (its first observed patch, with
 * {@code attend_missing_patches}); time positions count patches from the first patch any row of
 * the series observes.
 */
public final class BlockLayout {

    private final int rows;
    private final int patch;
    private final int contextPatches;
    private final int encoder;
    private final int horizon;
    private final boolean covariates;
    private final int[] group;
    private final int[] role;
    private final boolean[][] keyValid;
    private final int[] first;
    private final float[][] window;
    private final boolean[][] observed;

    private BlockLayout(int rows, int patch, int contextPatches, int encoder, int horizon, boolean covariates, int[] group,
            int[] role, boolean[][] keyValid, int[] first, float[][] window, boolean[][] observed) {
        this.rows = rows;
        this.patch = patch;
        this.contextPatches = contextPatches;
        this.encoder = encoder;
        this.horizon = horizon;
        this.covariates = covariates;
        this.group = group;
        this.role = role;
        this.keyValid = keyValid;
        this.first = first;
        this.window = window;
        this.observed = observed;
    }

    public static BlockLayout of(TimestralConfig config, BlockInput b) {
        int ps = config.patchSize();
        int r = b.rows();
        int t = b.length();
        if (t % ps != 0) {
            throw new IllegalArgumentException("context length " + t + " is not a multiple of the patch size " + ps);
        }
        int h = b.horizonPatches();
        if (h < 1) {
            throw new IllegalArgumentException("horizon patches must be >= 1, got " + h);
        }
        boolean cov = false;
        if (b.role() != null && config.covariateEnabled()) {
            for (int x : b.role()) {
                cov |= x != BlockInput.TARGET;
            }
        }
        int ll = t / ps;
        float[][] window = b.inputs();
        boolean[][] obs = b.observed();
        int[] role = new int[r];
        if (cov) {
            if (h > config.blockSize()) {
                throw new IllegalArgumentException("a block with covariates decodes at most " + config.blockSize()
                        + " patches, got " + h);
            }
            System.arraycopy(b.role(), 0, role, 0, r);
            window = new float[r][t + h * ps];
            obs = new boolean[r][t + h * ps];
            for (int i = 0; i < r; i++) {
                System.arraycopy(b.inputs()[i], 0, window[i], 0, t);
                System.arraycopy(b.observed()[i], 0, obs[i], 0, t);
                if (b.knownFuture() != null) {
                    System.arraycopy(b.knownFuture()[i], 0, window[i], t, h * ps);
                    System.arraycopy(b.knownFutureMask()[i], 0, obs[i], t, h * ps);
                }
            }
        }
        int enc = window[0].length / ps;
        boolean[][] valid = new boolean[r][enc];
        boolean[][] patchObserved = new boolean[r][enc];
        for (int i = 0; i < r; i++) {
            boolean started = false;
            for (int p = 0; p < enc; p++) {
                boolean any = false;
                for (int s = p * ps; s < (p + 1) * ps; s++) {
                    any |= obs[i][s];
                }
                patchObserved[i][p] = any;
                started |= any;
                valid[i][p] = config.attendMissingPatches() ? started : any;
            }
        }
        // the first patch any row of the series observes (0 if none does)
        Map<Integer, Integer> groupFirst = new HashMap<>();
        for (int i = 0; i < r; i++) {
            int f = enc;
            for (int p = 0; p < enc; p++) {
                if (patchObserved[i][p]) {
                    f = p;
                    break;
                }
            }
            groupFirst.merge(b.group()[i], f, Math::min);
        }
        int[] first = new int[r];
        for (int i = 0; i < r; i++) {
            int f = groupFirst.get(b.group()[i]);
            first[i] = f == enc ? 0 : f;
        }
        return new BlockLayout(r, ps, ll, enc, h, cov, b.group(), role, valid, first, window, obs);
    }

    public int rows() {
        return rows;
    }

    /** {@code ll}: the context patches; the horizon starts after them. */
    public int contextPatches() {
        return contextPatches;
    }

    /** Encoder tokens per row: the context patches, plus the horizon's with covariates. */
    public int encoderTokens() {
        return encoder;
    }

    public int horizonPatches() {
        return horizon;
    }

    /** Encoder plus decoder tokens per row. */
    public int tokens() {
        return encoder + horizon;
    }

    /** Whether this block runs the covariate path (some row is a covariate). */
    public boolean covariates() {
        return covariates;
    }

    /** {@code [rows][encoderTokens · P]}: the context, plus the known-future values with covariates. */
    public float[][] window() {
        return window;
    }

    public boolean[][] observed() {
        return observed;
    }

    public int group(int row) {
        return group[row];
    }

    public int role(int row) {
        return role[row];
    }

    public boolean known(int row) {
        return covariates && role[row] == BlockInput.KNOWN;
    }

    public boolean keyValid(int row, int patch) {
        return keyValid[row][patch];
    }

    /** The time position of token {@code token} of a row (encoder tokens first, then decoder tokens). */
    public int position(int row, int token) {
        int p = token < encoder ? token : contextPatches + token - encoder;
        return p - first[row];
    }

    /** Whether, in time attention on row {@code row}, query token {@code q} may attend key token {@code k}. */
    public boolean timeAllowed(int row, int q, int k) {
        if (q < encoder) {
            if (k >= encoder) {
                return false;
            }
            return k == q || keyValid[row][k] && (k <= q || known(row));
        }
        if (k < encoder) {
            return k < contextPatches && keyValid[row][k];
        }
        return k <= q;
    }

    /** Whether, in variate attention at token slot {@code slot}, row {@code q} may attend row {@code k}. */
    public boolean variateAllowed(int slot, int q, int k) {
        if (group[q] != group[k]) {
            return false;
        }
        if (q == k) {
            return true;
        }
        if (slot < encoder && !keyValid[k][slot]) {
            return false;
        }
        if (covariates) {
            if (role[q] != BlockInput.TARGET && role[k] == BlockInput.TARGET) {
                return false;
            }
            if (role[q] == BlockInput.KNOWN && role[k] != BlockInput.KNOWN) {
                return false;
            }
        }
        return true;
    }

    /** The patch size. */
    public int patch() {
        return patch;
    }
}
