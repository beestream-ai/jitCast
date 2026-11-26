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
package ai.beestream.jitcast.timestral.tornado;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.timestral.BlockInput;
import ai.beestream.jitcast.timestral.TimestralGolden;
import ai.beestream.jitcast.timestral.TimestralPipeline;
import ai.beestream.jitcast.timestral.TimestralWeights;
import ai.beestream.jitcast.timestral.cpu.CpuTimestral;
import org.junit.Test;

/**
 * The TornadoVM {@code predict_block} against the CPU reference on a large mixed batch the golden
 * cases do not cover: 37 series of different lengths, with gaps and a late start, some with past or
 * known-future covariates, so the attention tiles span several series and a series spans tiles.
 *
 * <p>In FP32 every call's raw output must match. With reduced-precision GEMM inputs only the forecasts
 * are checked: a block's raw output is the {@code sinh} of the model's prediction, so its top quantiles
 * grow exponentially along the horizon (to 1e11 and beyond on these random walks), and a small error
 * in the model's own space becomes a large one there. The pipeline caps those values to the observed
 * range before they reach a forecast.
 *
 * <p>The forecasts are compared on the series with at least {@link #MIN_OBSERVED} observed steps. The
 * batch also holds series with 36 to 86 observed steps forecast 1,100 steps ahead: there the model's
 * far quantiles sit near the FP32 range, and with TF32 or FP16 GEMM inputs the smaller checkpoints move
 * them far enough that the reference's non-finite clamp and the cap give a different forecast. Their
 * raw outputs are still checked in FP32.
 *
 * <p>The outer quantiles far into the horizon amplify any difference in the GEMMs' rounding: in FP32
 * the forecasts are held to {@link #FORECAST_TOLERANCE} (Timestral-5M reaches 1.9e-4 against raw calls
 * within 3e-5), and with reduced-precision GEMM inputs only the median is held to the bound, as a
 * fraction of the forecast's largest value over all quantiles (Timestral-5M's outer quantiles move by
 * up to 19% of it there).
 */
public class TimestralBatchParityAccelTest {

    private static final double TOLERANCE = 1e-4;
    /** FP32 forecasts, every quantile. */
    private static final double FORECAST_TOLERANCE = 1e-3;
    /**
     * With FP16 or TF32 GEMM inputs ({@code -Dtimestral.tornado.gemm=half|cublas-fp16|cublas-tf32}): a
     * bound for a smoke check, not parity, on each series' forecast.
     */
    private static final double REDUCED_TOLERANCE = 5e-2;
    /** Series with fewer observed target steps are left out of the forecast comparison. */
    private static final int MIN_OBSERVED = 128;

    @Test
    public void mixedBatchMatchesTheCpuReference() throws IOException {
        try (TimestralWeights weights = TimestralWeights.load(TimestralGolden.checkpoint());
                TimestralForward fwd = new TimestralForward(weights.config(), TimestralPacked.pack(weights))) {
            CpuTimestral cpu = new CpuTimestral(weights);
            List<BlockInput> calls = new ArrayList<>();
            int horizon = 1100;
            TimestralPipeline pipeline = new TimestralPipeline(weights.config(), b -> {
                calls.add(b);
                return cpu.predictBlock(b);
            });
            List<Forecast> expected = pipeline.forecast(batch(new Random(11), 37, horizon), horizon);
            List<String> failures = new ArrayList<>();
            boolean reduced = fwd.gemmMode().reducedPrecision();
            double limit = reduced ? REDUCED_TOLERANCE : TOLERANCE;
            for (int k = 0; k < calls.size(); k++) {
                BlockInput b = calls.get(k);
                double err = TimestralGolden.relativeMaxError(flat(fwd.predictBlock(b)), flat(cpu.predictBlock(b)));
                System.out.printf("mixed batch call%d rows=%d length=%d horizon patches=%d covariates=%b  rel max err %.2e%n", k,
                        b.rows(), b.length(), b.horizonPatches(), b.role() != null, err);
                if (!reduced && !(err <= limit)) {
                    failures.add("call" + k + ": " + err);
                }
            }
            List<TimeSeries> series = batch(new Random(11), 37, horizon);
            List<Forecast> actual = new TimestralPipeline(weights.config(), fwd::predictBlock).forecast(series, horizon);
            double forecastLimit = reduced ? REDUCED_TOLERANCE : FORECAST_TOLERANCE;
            int median = weights.config().medianIndex();
            double worst = 0, worstMedian = 0;
            int compared = 0;
            for (int s = 0; s < expected.size(); s++) {
                if (observed(series.get(s)) < MIN_OBSERVED) {
                    continue;
                }
                compared++;
                float[][][] a = actual.get(s).values(), e = expected.get(s).values();
                double err = TimestralGolden.relativeMaxError(flat(a), flat(e));
                double medianErr = medianError(a, e, median);
                worst = Math.max(worst, err);
                worstMedian = Math.max(worstMedian, medianErr);
                double checked = reduced ? medianErr : err;
                if (!(checked <= forecastLimit)) {
                    failures.add("series" + s + (reduced ? " median" : "") + " forecast: " + checked);
                }
            }
            System.out.printf("mixed batch forecasts, %d of %d series  rel max err %.2e, median %.2e%n", compared, expected.size(),
                    worst, worstMedian);
            assertTrue(String.join("\n", failures), failures.isEmpty());
        }
    }

    /** The median's largest difference over the forecast's largest magnitude, all quantiles. */
    private static double medianError(float[][][] actual, float[][][] expected, int median) {
        double diff = 0, scale = 1e-6;
        for (int t = 0; t < expected.length; t++) {
            for (int k = 0; k < expected[t].length; k++) {
                for (int h = 0; h < expected[t][k].length; h++) {
                    scale = Math.max(scale, Math.abs(expected[t][k][h]));
                    if (k == median) {
                        diff = Math.max(diff, Math.abs((double) actual[t][k][h] - expected[t][k][h]));
                    }
                }
            }
        }
        return diff / scale;
    }

    private static int observed(TimeSeries series) {
        int n = 0;
        for (float x : series.target()[0]) {
            if (!Float.isNaN(x)) {
                n++;
            }
        }
        return n;
    }

    static List<TimeSeries> batch(Random rnd, int count, int horizon) {
        List<TimeSeries> out = new ArrayList<>();
        for (int s = 0; s < count; s++) {
            int len = 40 + rnd.nextInt(900);
            int targets = 1 + (s % 5 == 0 ? 1 : 0);
            float[][] t = walk(rnd, targets, len);
            if (s % 7 == 3) {
                for (int i = 0; i < Math.min(len, 50); i++) {
                    t[0][i] = Float.NaN; // a late start
                }
            }
            if (s % 4 == 1) {
                for (int i = len / 3; i < len / 3 + 20 && i < len; i++) {
                    t[0][i] = Float.NaN; // a gap
                }
            }
            float[][] past = s % 3 == 1 ? walk(rnd, 1 + rnd.nextInt(2), len) : null;
            float[][] known = null;
            if (s % 3 == 2) {
                known = new float[1 + rnd.nextInt(2)][len + horizon];
                for (float[] row : known) {
                    for (int i = 0; i < row.length; i++) {
                        row[i] = (i % 24) < 6 ? 1f : 0f;
                    }
                }
            }
            out.add(new TimeSeries(t, past, known));
        }
        return out;
    }

    private static float[][] walk(Random rnd, int rows, int len) {
        float[][] r = new float[rows][len];
        for (float[] row : r) {
            float v = 10f * rnd.nextFloat();
            for (int i = 0; i < len; i++) {
                v += (float) rnd.nextGaussian() * 0.3f;
                row[i] = v + 2f * (float) Math.sin(2 * Math.PI * i / 24.0);
            }
        }
        return r;
    }

    private static float[] flat(float[][][] v) {
        int q = v[0][0].length;
        float[] f = new float[v.length * v[0].length * q];
        int i = 0;
        for (float[][] row : v) {
            for (float[] x : row) {
                System.arraycopy(x, 0, f, i, q);
                i += q;
            }
        }
        return f;
    }
}
