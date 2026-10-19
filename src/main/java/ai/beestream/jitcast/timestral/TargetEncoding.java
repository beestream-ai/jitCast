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

import java.util.Arrays;

import ai.beestream.jitcast.api.TimeSeries;

/**
 * Categorical covariates turned into numeric ones before they reach the model, as
 * {@code timestral.covariate_encoding} and {@code TimestralForecaster._encode_categorical} do.
 *
 * <p>In a series with one target, category {@code c} becomes the smoothed mean of the target over
 * the context where it occurs, {@code (mean + sum_c) / (1 + count_c)}, against the forward-filled
 * target from its first observation on, within the last {@code maxContext} steps. A missing code
 * ({@code NaN}) is a category of its own, and a code the context never shows (in the horizon of a
 * known covariate) becomes the series mean. A series with several targets has no single target to
 * encode against, so its codes are used as they are.
 */
public final class TargetEncoding {

    private static final double SMOOTH = 1.0;

    private TargetEncoding() {}

    /**
     * The series with its categorical covariates encoded and appended to its numeric ones: the
     * encoded past rows after the numeric past covariates, the encoded known rows after the numeric
     * future covariates, as the reference stacks them.
     */
    public static TimeSeries apply(TimeSeries ts, int horizon, int maxContext) {
        if (!ts.hasCategorical()) {
            return ts;
        }
        int len = ts.length();
        double[] target = ts.targets() == 1 ? encodingTarget(ts.target()[0], maxContext) : null;
        float[][] past = new float[ts.pastCategoricalCount()][];
        for (int c = 0; c < past.length; c++) {
            past[c] = encodeRow(target, ts.pastCategorical()[c], len, 0);
        }
        float[][] known = new float[ts.futureCategoricalCount()][];
        for (int c = 0; c < known.length; c++) {
            double[] codes = ts.futureCategorical()[c];
            if (codes.length < len + horizon) {
                throw new IllegalArgumentException("future categorical covariates need at least " + (len + horizon)
                        + " steps, got " + codes.length);
            }
            known[c] = encodeRow(target, codes, len, horizon);
        }
        return new TimeSeries(ts.target(), concat(ts.pastCovariates(), past), concat(ts.futureCovariates(), known));
    }

    /**
     * {@code _encoding_target}: the forward-filled target from its first observation on, within the
     * last {@code maxContext} steps, and {@code NaN} elsewhere.
     */
    static double[] encodingTarget(float[] row, int maxContext) {
        int t = row.length;
        double[] out = new double[t];
        boolean started = false;
        double last = 0;
        for (int i = 0; i < t; i++) {
            boolean observed = Float.isFinite(row[i]);
            if (observed) {
                last = row[i];
            }
            started |= observed;
            out[i] = started && i >= t - maxContext ? last : Double.NaN;
        }
        return out;
    }

    /**
     * One categorical row: the first {@code len} codes (the context) encoded, then {@code horizon}
     * codes of the horizon; the raw codes as floats without a target to encode against.
     */
    static float[] encodeRow(double[] target, double[] codes, int len, int horizon) {
        float[] out = new float[len + horizon];
        if (target == null) {
            for (int i = 0; i < out.length; i++) {
                out[i] = (float) codes[i];
            }
            return out;
        }
        double[] past = Arrays.copyOf(codes, len);
        double[] future = horizon > 0 ? Arrays.copyOfRange(codes, len, len + horizon) : null;
        float[][] enc = encode(target, past, future);
        for (int i = 0; i < len; i++) {
            // a missing code before the target starts is padding, not the missing category
            out[i] = Double.isNaN(target[i]) && Double.isNaN(past[i]) ? Float.NaN : enc[0][i];
        }
        if (future != null) {
            System.arraycopy(enc[1], 0, out, len, horizon);
        }
        return out;
    }

    /**
     * {@code target_encode}: per-series smoothed mean-target encoding of {@code past} (aligned with
     * {@code target}) and, if given, of {@code future}. Returns {@code {past, future}} (future null when
     * not given), rounded to float as the reference returns them.
     */
    static float[][] encode(double[] target, double[] past, double[] future) {
        int t = target.length;
        double count = 0, sum = 0;
        for (double v : target) {
            if (Double.isFinite(v)) {
                count++;
                sum += v;
            }
        }
        double mean = count > 0 ? sum / count : 0.0;
        double[] categories = Arrays.stream(past).filter(Double::isFinite).distinct().sorted().toArray();
        int nanSlot = categories.length;
        double[] sums = new double[nanSlot + 1], counts = new double[nanSlot + 1];
        int[] slots = new int[t];
        for (int i = 0; i < t; i++) {
            slots[i] = slot(past[i], categories, nanSlot);
            if (slots[i] < 0) {
                throw new IllegalStateException("every past code must be a category");
            }
            if (Double.isFinite(target[i])) {
                sums[slots[i]] += target[i];
                counts[slots[i]] += 1;
            }
        }
        double[] lookup = new double[nanSlot + 2];
        for (int s = 0; s <= nanSlot; s++) {
            lookup[s] = (SMOOTH * mean + sums[s]) / (SMOOTH + counts[s]);
        }
        lookup[nanSlot + 1] = mean; // a code the context never shows
        float[] encPast = new float[t];
        for (int i = 0; i < t; i++) {
            encPast[i] = (float) lookup[slots[i]];
        }
        if (future == null) {
            return new float[][] {encPast, null};
        }
        float[] encFuture = new float[future.length];
        for (int i = 0; i < future.length; i++) {
            int s = slot(future[i], categories, nanSlot);
            encFuture[i] = (float) lookup[s < 0 ? nanSlot + 1 : s];
        }
        return new float[][] {encPast, encFuture};
    }

    /** The slot of a code: its index among the categories, {@code nanSlot} for NaN, −1 if unseen. */
    private static int slot(double code, double[] categories, int nanSlot) {
        if (Double.isNaN(code)) {
            return nanSlot;
        }
        int i = Arrays.binarySearch(categories, code);
        return i >= 0 ? i : -1;
    }

    private static float[][] concat(float[][] a, float[][] b) {
        if (b.length == 0) {
            return a;
        }
        if (a == null || a.length == 0) {
            return b;
        }
        float[][] r = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }
}
