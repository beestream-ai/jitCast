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
import java.util.TreeSet;

/**
 * The reference's quantile-level math ({@code t0.levels}, {@code t0.quantile}): linear interpolation between
 * the trained levels, exponential tails beyond them, and the weighted-quantile reduction of the
 * autoregressive rollout's trajectories. Arrays are row-major with the levels last, in float32 as in the
 * reference.
 */
public final class T0Quantiles {

    private T0Quantiles() {}

    /**
     * {@return the requested levels, checked}
     *
     * @param levels the levels a forecast asks for
     * @throws IllegalArgumentException unless they are ascending, unique and in (0, 1)
     */
    public static double[] validate(double[] levels) {
        if (levels.length == 0) {
            throw new IllegalArgumentException("quantile levels must be non-empty");
        }
        for (int i = 0; i < levels.length; i++) {
            if (!(levels[i] > 0 && levels[i] < 1)) {
                throw new IllegalArgumentException("each quantile level must be in (0, 1); got " + levels[i]);
            }
            if (i > 0 && !(levels[i] > levels[i - 1])) {
                throw new IllegalArgumentException("quantile levels must be ascending without duplicates");
            }
        }
        return levels.clone();
    }

    /** The trained levels rounded to 6 decimals, as {@code select_tail_knots} compares them. */
    private static double round6(double v) {
        return new java.math.BigDecimal(v).setScale(6, java.math.RoundingMode.HALF_EVEN).doubleValue();
    }

    /**
     * {@return the trained levels the exponential tails pin through, or null when no requested level is
     * beyond the trained range} ({@code select_tail_knots})
     *
     * @param trained the trained levels
     * @param requested the requested levels
     */
    public static double[] tailKnots(double[] trained, double[] requested) {
        double[] ordered = new double[trained.length];
        for (int i = 0; i < trained.length; i++) {
            ordered[i] = round6(trained[i]);
        }
        Arrays.sort(ordered);
        double lo = ordered[0], hi = ordered[ordered.length - 1];
        boolean any = false;
        for (double q : requested) {
            any |= (q < lo || q > hi) && q > 0 && q < 1;
        }
        if (!any) {
            return null;
        }
        if (ordered.length < 2) {
            throw new IllegalArgumentException("need at least two trained quantile levels to pin a tail");
        }
        TreeSet<Double> k = new TreeSet<>(List.of(ordered[0], ordered[1], ordered[ordered.length - 2], ordered[ordered.length - 1]));
        return k.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * {@return the levels a rollout produces to serve a request} ({@code get_rollout_quantile_levels})
     *
     * @param trained the trained levels
     * @param requested the requested levels
     */
    public static double[] rolloutLevels(double[] trained, double[] requested) {
        double[] knots = tailKnots(trained, requested);
        if (knots == null) {
            return requested.clone();
        }
        TreeSet<Double> out = new TreeSet<>();
        for (double q : requested) {
            if (knots[0] <= q && q <= knots[knots.length - 1]) {
                out.add(q);
            }
        }
        for (double k : knots) {
            out.add(k);
        }
        return out.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * {@code column[lower] + weight · (column[upper] − column[lower])} per query level.
     *
     * @param lower the lower source column of each query
     * @param upper the upper source column of each query
     * @param weight the weight of each query, in float32
     */
    record Plan(int[] lower, int[] upper, float[] weight) {}

    /** {@code interpolation_plan}: brackets among the source levels padded with 0 and 1, in float32. */
    static Plan interpolationPlan(double[] query, double[] source) {
        int n = source.length;
        boolean padLeft = (float) source[0] > 0, padRight = (float) source[n - 1] < 1;
        float[] padded = new float[n + (padLeft ? 1 : 0) + (padRight ? 1 : 0)];
        int o = 0;
        if (padLeft) {
            padded[o++] = 0f;
        }
        for (double s : source) {
            padded[o++] = (float) s;
        }
        if (padRight) {
            padded[o] = 1f;
        }
        int[] lower = new int[query.length], upper = new int[query.length];
        float[] weight = new float[query.length];
        for (int i = 0; i < query.length; i++) {
            float q = (float) query[i];
            int up = 0;
            while (up < padded.length && padded[up] <= q) {
                up++;
            }
            up = Math.min(up, padded.length - 1);
            int lo = up - 1;
            float span = padded[up] - padded[lo];
            int shift = padLeft ? 1 : 0;
            lower[i] = Math.min(Math.max(lo - shift, 0), n - 1);
            upper[i] = Math.min(Math.max(up - shift, 0), n - 1);
            weight[i] = span == 0f ? 0f : (q - padded[lo]) / span;
        }
        return new Plan(lower, upper, weight);
    }

    /**
     * {@return the query levels linearly interpolated between the source levels, the edge levels held flat
     * beyond them} ({@code interpolate_quantiles})
     *
     * @param query the levels to return
     * @param source the levels of {@code values}
     * @param values {@code [n · source]}
     */
    public static float[] interpolate(double[] query, double[] source, float[] values) {
        Plan p = interpolationPlan(query, source);
        int s = source.length, q = query.length, n = values.length / s;
        float[] out = new float[n * q];
        for (int r = 0; r < n; r++) {
            for (int i = 0; i < q; i++) {
                float lo = values[r * s + p.lower[i]], up = values[r * s + p.upper[i]];
                out[r * q + i] = lo + p.weight[i] * (up - lo);
            }
        }
        return out;
    }

    /**
     * {@return the query levels, those beyond the trained range evaluated on exponential tails pinned through
     * the outermost trained levels} ({@code extrapolate_quantiles})
     *
     * @param query the levels to return
     * @param original the levels of {@code values}, as {@link #rolloutLevels} returned them
     * @param values {@code [n · original]}
     * @param trained the trained levels
     */
    public static float[] extrapolate(double[] query, double[] original, float[] values, double[] trained) {
        double[] knots = tailKnots(trained, query);
        if (knots == null) {
            return values;
        }
        List<Double> left = new ArrayList<>(), right = new ArrayList<>();
        for (double q : query) {
            if (q < knots[0]) {
                left.add(q);
            } else if (q > knots[knots.length - 1]) {
                right.add(q);
            }
        }
        List<Double> full = new ArrayList<>(left);
        for (double o : original) {
            full.add(o);
        }
        full.addAll(right);
        int[] knotColumns = new int[knots.length];
        for (int i = 0; i < knots.length; i++) {
            knotColumns[i] = indexOf(original, knots[i]);
        }
        int[] order = new int[query.length];
        for (int i = 0; i < query.length; i++) {
            order[i] = full.indexOf(query[i]);
        }
        // The knots and tail levels as the float32 values the arrays hold; the log-ratios in float64.
        double[] k = new double[knots.length];
        for (int i = 0; i < k.length; i++) {
            k[i] = (float) knots[i];
        }
        int kn = k.length, nOrig = original.length, n = values.length / nOrig, nl = left.size(), nr = right.size();
        float[] leftRatio = new float[nl], rightRatio = new float[nr];
        for (int i = 0; i < nl; i++) {
            leftRatio[i] = (float) Math.log((float) (double) left.get(i) / k[0]);
        }
        for (int i = 0; i < nr; i++) {
            rightRatio[i] = (float) Math.log((1.0 - k[kn - 1]) / (1.0 - (float) (double) right.get(i)));
        }
        float leftDiv = (float) Math.log(k[1] / k[0]);
        float rightDiv = (float) Math.log((1.0 - k[kn - 2]) / (1.0 - k[kn - 1]));
        float[] out = new float[n * query.length];
        float[] row = new float[nl + nOrig + nr];
        for (int r = 0; r < n; r++) {
            float v0 = values[r * nOrig + knotColumns[0]], v1 = values[r * nOrig + knotColumns[1]];
            float vn1 = values[r * nOrig + knotColumns[kn - 2]], vn = values[r * nOrig + knotColumns[kn - 1]];
            float sl = Math.max((v1 - v0) / leftDiv, 0f), sr = Math.max((vn - vn1) / rightDiv, 0f);
            for (int i = 0; i < nl; i++) {
                row[i] = v0 + sl * leftRatio[i];
            }
            System.arraycopy(values, r * nOrig, row, nl, nOrig);
            for (int i = 0; i < nr; i++) {
                row[nl + nOrig + i] = vn + sr * rightRatio[i];
            }
            for (int i = 0; i < query.length; i++) {
                out[r * query.length + i] = row[order[i]];
            }
        }
        return out;
    }

    private static int indexOf(double[] a, double v) {
        for (int i = 0; i < a.length; i++) {
            if (a[i] == v) {
                return i;
            }
        }
        throw new IllegalStateException("level " + v + " not among " + Arrays.toString(a));
    }

    /** {@code get_prob_mass_per_quantile_level}: trapezoidal masses, normalized, in float32. */
    static float[] probabilityMass(double[] levels) {
        int n = levels.length;
        float[] b = new float[n + 2];
        b[n + 1] = 1f;
        for (int i = 0; i < n; i++) {
            b[i + 1] = (float) levels[i];
        }
        float[] m = new float[n];
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            m[i] = (b[i + 2] - b[i]) / 2f;
            sum += m[i];
        }
        for (int i = 0; i < n; i++) {
            m[i] /= sum;
        }
        return m;
    }

    /**
     * {@return the rollout's trajectories reduced to the query levels} ({@code reduce_rollout_quantiles}):
     * every (trajectory, trained level) pair is a sample weighted by both levels' probability masses, and the
     * query levels are read off each step's weighted empirical distribution.
     *
     * @param block {@code [targets · paths · trained · horizon]}: each target's trajectory per query level
     * @param targets the number of targets
     * @param horizon the steps
     * @param trained the trained levels
     * @param query the query levels, one trajectory each
     * @return {@code [targets · horizon · query]}
     */
    public static float[] reduceRollout(float[] block, int targets, int horizon, double[] trained, double[] query) {
        int np = trained.length, nq = query.length, ns = np * nq;
        float[] mp = probabilityMass(trained), mq = probabilityMass(query);
        float[] w = new float[ns];
        float wsum = 0f;
        for (int i = 0; i < np; i++) {
            for (int j = 0; j < nq; j++) {
                w[i * nq + j] = mp[i] * mq[j];
            }
        }
        for (float v : w) {
            wsum += v;
        }
        for (int i = 0; i < ns; i++) {
            w[i] = w[i] / wsum;
        }
        float[] qf = new float[nq];
        for (int j = 0; j < nq; j++) {
            qf[j] = (float) query[j];
        }
        float[] out = new float[targets * horizon * nq];
        float[] samples = new float[ns];
        Integer[] order = new Integer[ns];
        float[] levels = new float[ns + 2], vals = new float[ns + 2];
        for (int t = 0; t < targets; t++) {
            for (int h = 0; h < horizon; h++) {
                // sample (i, j): trained level i of the trajectory of query level j
                for (int i = 0; i < np; i++) {
                    for (int j = 0; j < nq; j++) {
                        samples[i * nq + j] = block[((t * nq + j) * np + i) * horizon + h];
                    }
                }
                for (int i = 0; i < ns; i++) {
                    order[i] = i;
                }
                Arrays.sort(order, (a, b) -> Float.compare(samples[a], samples[b]));
                double cdf = 0;
                levels[0] = 0f;
                for (int i = 0; i < ns; i++) {
                    cdf += w[order[i]];
                    levels[i + 1] = Math.min(Math.max((float) cdf, 0f), 1f);
                    vals[i + 1] = samples[order[i]];
                }
                levels[ns + 1] = 1f;
                vals[0] = vals[1];
                vals[ns + 1] = vals[ns];
                for (int j = 0; j < nq; j++) {
                    int up = 0;
                    for (int i = 0; i < ns + 2; i++) {
                        up += levels[i] <= qf[j] ? 1 : 0;
                    }
                    up = Math.min(up, ns + 1);
                    int lo = up - 1;
                    float weight = (qf[j] - levels[lo]) / (levels[up] - levels[lo]);
                    if (Float.isNaN(weight)) {
                        weight = 0f;
                    }
                    out[(t * horizon + h) * nq + j] = vals[lo] + weight * (vals[up] - vals[lo]);
                }
            }
        }
        return out;
    }
}
