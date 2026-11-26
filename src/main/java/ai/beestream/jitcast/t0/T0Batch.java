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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import ai.beestream.jitcast.api.TimeSeries;

/**
 * t0's native model input, the reference's {@code TimeSeries}: one row per variate of every series, all on
 * one time axis, with a value, a mask, a group id and a variate type per cell. Rows of one series share a
 * group id; cells that only pad a shorter series carry group and type {@code -1}.
 *
 * @param rows the number of rows
 * @param width the time steps per row
 * @param values {@code [rows · width]} the values, 0 wherever there is no observation
 * @param mask {@code [rows · width]} the {@code MaskType} of each cell
 * @param group {@code [rows · width]} the group id of each cell
 * @param type {@code [rows · width]} the {@code VariateType} of each cell
 */
public record T0Batch(int rows, int width, float[] values, byte[] mask, int[] group, int[] type) {

    /** {@code MaskType.VALID}: an observation. */
    public static final byte VALID = 0;
    /** {@code MaskType.PAD}: a cell that only pads a row out to the batch's width. */
    public static final byte PAD = 1;
    /** {@code MaskType.MISSING}: an absent observation. */
    public static final byte MISSING = 2;
    /** {@code MaskType.WITHHELD}: a step to predict. */
    public static final byte WITHHELD = 4;

    /** {@code VariateType.TARGET}. */
    public static final int TARGET = 0;
    /** {@code VariateType.HISTORICAL}: a covariate known over the past only. */
    public static final int HISTORICAL = 1;
    /** {@code VariateType.FUTURE}: a covariate known over the past and the horizon. */
    public static final int FUTURE = 2;

    /**
     * {@return the batch of a list of series, as {@code TimeSeries.batch} builds it from each series'
     * {@code TimeSeries} over its context and the horizon} Each series' rows are its targets, its past-only
     * covariates (historical, withheld over the horizon like the targets) and its known-future covariates
     * (cut to context + horizon), right-aligned to the longest series.
     *
     * @param series the inputs
     * @param horizon the forecast steps
     * @throws IllegalArgumentException if a series has categorical covariates or a future covariate shorter
     *     than its context plus the horizon
     */
    public static T0Batch of(List<TimeSeries> series, int horizon) {
        int width = 0, rows = 0;
        for (TimeSeries ts : series) {
            if (ts.hasCategorical()) {
                throw new IllegalArgumentException("t0 does not take categorical covariates");
            }
            for (int f = 0; f < ts.futureCount(); f++) {
                if (ts.futureCovariates()[f].length < ts.length() + horizon) {
                    throw new IllegalArgumentException("a future covariate must cover the context and the horizon ("
                            + (ts.length() + horizon) + " steps), got " + ts.futureCovariates()[f].length);
                }
            }
            width = Math.max(width, ts.length() + horizon);
            rows += ts.rows();
        }
        T0Batch b = empty(rows, width);
        int[] first = new int[series.size()];
        for (int s = 1; s < series.size(); s++) {
            first[s] = first[s - 1] + series.get(s - 1).rows();
        }
        int w = width;
        IntStream all = IntStream.range(0, series.size());
        // series are independent: fill them in parallel when there is enough work
        (series.size() > 1 && (long) rows * width >= T0Scaler.PARALLEL_CELLS ? all.parallel() : all).forEach(s -> {
            TimeSeries ts = series.get(s);
            int t = ts.length(), start = w - t - horizon, r = first[s];
            for (int kind = TARGET; kind <= FUTURE; kind++) {
                float[][] src = kind == TARGET ? ts.target() : kind == HISTORICAL ? ts.pastCovariates() : ts.futureCovariates();
                if (src == null) {
                    continue;
                }
                for (float[] row : src) {
                    int o = r * w + start;
                    for (int i = 0; i < t + horizon; i++) {
                        boolean known = kind == FUTURE || i < t;
                        float v = known ? row[i] : 0f;
                        b.values[o + i] = Float.isNaN(v) ? 0f : v;
                        b.mask[o + i] = !known ? WITHHELD : Float.isNaN(v) ? MISSING : VALID;
                        b.group[o + i] = s;
                        b.type[o + i] = kind;
                    }
                    r++;
                }
            }
        });
        return b;
    }

    /**
     * {@return a batch of padding cells: value 0, {@code PAD}, group and type {@code -1}}
     *
     * @param rows the rows
     * @param width the time steps
     */
    public static T0Batch empty(int rows, int width) {
        byte[] mask = new byte[rows * width];
        int[] group = new int[rows * width], type = new int[rows * width];
        Arrays.fill(mask, PAD);
        Arrays.fill(group, -1);
        Arrays.fill(type, -1);
        return new T0Batch(rows, width, new float[rows * width], mask, group, type);
    }

    /** {@return a deep copy} */
    public T0Batch copy() {
        return new T0Batch(rows, width, values.clone(), mask.clone(), group.clone(), type.clone());
    }

    /**
     * {@return the type of a row: its type at the last time step}
     *
     * @param r the row
     */
    public int rowType(int r) {
        return type[r * width + width - 1];
    }

    /**
     * {@return the group of a row: its group at the last time step}
     *
     * @param r the row
     */
    public int rowGroup(int r) {
        return group[r * width + width - 1];
    }

    /** {@return the indices of the target rows, in order} */
    public int[] targetRows() {
        int n = 0;
        int[] out = new int[rows];
        for (int r = 0; r < rows; r++) {
            if (rowType(r) == TARGET) {
                out[n++] = r;
            }
        }
        return Arrays.copyOf(out, n);
    }

    /** {@return whether any row is a known-future covariate} */
    public boolean hasFuture() {
        for (int r = 0; r < rows; r++) {
            if (rowType(r) == FUTURE) {
                return true;
            }
        }
        return false;
    }

    /** {@return where the forecast region begins: the first step withheld on a target row, else the width} */
    public int contextEnd() {
        // the earliest withheld step of any target row, each row scanned in memory order
        int end = width;
        for (int r = 0; r < rows; r++) {
            if (rowType(r) != TARGET) {
                continue;
            }
            int o = r * width;
            for (int t = 0; t < end; t++) {
                if (mask[o + t] == WITHHELD) {
                    end = t;
                    break;
                }
            }
        }
        return end;
    }

    /** Whether a cell is an observation: {@code VALID} and finite. */
    boolean observed(int i) {
        return mask[i] == VALID && Float.isFinite(values[i]);
    }

    /**
     * {@return the window of time steps {@code [start, stop)}}
     *
     * @param start the first step
     * @param stop the step after the last
     */
    public T0Batch slice(int start, int stop) {
        int w = stop - start;
        T0Batch out = empty(rows, w);
        for (int r = 0; r < rows; r++) {
            System.arraycopy(values, r * width + start, out.values, r * w, w);
            System.arraycopy(mask, r * width + start, out.mask, r * w, w);
            System.arraycopy(group, r * width + start, out.group, r * w, w);
            System.arraycopy(type, r * width + start, out.type, r * w, w);
        }
        return out;
    }

    /**
     * {@return the rows in {@code keep}, in that order}
     *
     * @param keep the row indices
     */
    public T0Batch select(int[] keep) {
        T0Batch out = empty(keep.length, width);
        for (int i = 0; i < keep.length; i++) {
            int r = keep[i];
            System.arraycopy(values, r * width, out.values, i * width, width);
            System.arraycopy(mask, r * width, out.mask, i * width, width);
            System.arraycopy(group, r * width, out.group, i * width, width);
            System.arraycopy(type, r * width, out.type, i * width, width);
        }
        return out;
    }

    /**
     * {@return the reference's {@code _trim_leading_missing}: every row of a series turned to padding before
     * its targets' first observation in the context}
     *
     * @param context the context length
     */
    public T0Batch trimLeadingMissing(int context) {
        T0Batch out = copy();
        List<Integer> groups = new ArrayList<>();
        for (int r = 0; r < rows; r++) {
            if (!groups.contains(rowGroup(r))) {
                groups.add(rowGroup(r));
            }
        }
        for (int g : groups) {
            int first = -1;
            for (int t = 0; t < context && first < 0; t++) {
                for (int r = 0; r < rows; r++) {
                    if (rowGroup(r) == g && rowType(r) == TARGET && observed(r * width + t)) {
                        first = t;
                        break;
                    }
                }
            }
            if (first < 0) {
                continue;
            }
            for (int r = 0; r < rows; r++) {
                if (rowGroup(r) != g) {
                    continue;
                }
                for (int t = 0; t < first; t++) {
                    int i = r * width + t;
                    if (out.mask[i] != PAD) {
                        out.mask[i] = PAD;
                        out.values[i] = 0f;
                    }
                }
            }
        }
        return out;
    }

    /**
     * {@return the reference's {@code _drop_constant_covariates}: the known-future covariate rows whose
     * observed context is empty or constant, removed}
     *
     * @param context the context length
     */
    public T0Batch dropConstantCovariates(int context) {
        int[] keep = new int[rows];
        int n = 0;
        for (int r = 0; r < rows; r++) {
            if (rowType(r) == FUTURE) {
                float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
                for (int t = 0; t < context; t++) {
                    int i = r * width + t;
                    if (observed(i)) {
                        min = Math.min(min, values[i]);
                        max = Math.max(max, values[i]);
                    }
                }
                if (!(min < max)) {
                    continue;
                }
            }
            keep[n++] = r;
        }
        return n == rows ? this : select(Arrays.copyOf(keep, n));
    }

    /**
     * {@return the reference's {@code _negated}: every row negated, its missing steps after its first
     * observation linearly interpolated (a gap past the last observation takes that value) and marked
     * observed}
     */
    public T0Batch negated() {
        T0Batch out = copy();
        for (int r = 0; r < rows; r++) {
            int o = r * width;
            int[] seen = new int[width];
            int n = 0;
            for (int t = 0; t < width; t++) {
                if (observed(o + t)) {
                    seen[n++] = t;
                }
            }
            double[] v = new double[width];
            for (int t = 0; t < width; t++) {
                v[t] = values[o + t];
            }
            if (n > 0) {
                int k = 0;
                for (int t = seen[0] + 1; t < width; t++) {
                    if (mask[o + t] != MISSING) {
                        continue;
                    }
                    while (k + 1 < n && seen[k + 1] <= t) {
                        k++;
                    }
                    if (k + 1 >= n) {
                        v[t] = values[o + seen[n - 1]];
                    } else {
                        int a = seen[k], b = seen[k + 1];
                        double va = values[o + a], vb = values[o + b];
                        v[t] = (vb - va) / (b - a) * (t - a) + va; // numpy.interp's order
                    }
                    out.mask[o + t] = VALID;
                }
            }
            for (int t = 0; t < width; t++) {
                out.values[o + t] = out.mask[o + t] != PAD ? (float) -v[t] : 0f;
            }
        }
        return out;
    }

    /**
     * {@return the observed context values of a row}
     *
     * @param r the row
     * @param context the context length
     */
    public double[] history(int r, int context) {
        double[] out = new double[context];
        int n = 0;
        for (int t = 0; t < context; t++) {
            if (observed(r * width + t)) {
                out[n++] = values[r * width + t];
            }
        }
        return Arrays.copyOf(out, n);
    }
}
