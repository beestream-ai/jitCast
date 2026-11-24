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
package ai.beestream.jitcast.t0.tornado;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;
import ai.beestream.jitcast.t0.T0Batch;
import ai.beestream.jitcast.t0.T0Config;
import ai.beestream.jitcast.t0.T0Forward;
import ai.beestream.jitcast.t0.T0Golden;
import ai.beestream.jitcast.t0.T0Pipeline;
import ai.beestream.jitcast.t0.T0Weights;
import ai.beestream.jitcast.t0.cpu.CpuT0;

/**
 * The TornadoVM forward pass against the reference: the quantile head of every recorded forward call's
 * output patches (on the reference's scaled window), and the whole pipeline's forecasts, for every golden
 * case; then a large mixed batch of every case's series against the CPU reference. FP32 GEMM modes must
 * match within {@code 1e-4}; modes with FP16 or TF32 GEMM inputs within {@link #REDUCED}.
 */
public class T0ForwardAccelTest {

    private static final double TOLERANCE = 1e-4;
    private static final double REDUCED = 5e-2;

    @Test
    public void headsAndForecastsMatchTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (T0Weights weights = T0Weights.load(T0Golden.checkpoint());
                TornadoT0Forward fwd = new TornadoT0Forward(weights.config(), T0Packed.pack(weights), 4)) {
            T0Config c = weights.config();
            int p = c.patchSize(), q = c.quantiles();
            double limit = fwd.gemmMode().reducedPrecision() ? REDUCED : TOLERANCE;
            System.out.println("GEMM mode " + fwd.gemmMode() + ", limit " + limit);
            for (String testCase : T0Golden.cases()) {
                Map<String, Object> info = T0Golden.caseInfo(testCase);
                List<?> horizons = (List<?>) info.get("horizons");
                try (SafeTensors g = T0Golden.open(testCase)) {
                    for (int k = 0; k < T0Golden.calls(g); k++) {
                        T0Batch w = T0Golden.window(g, k);
                        T0Forward.Input in = T0Forward.Input.of(w, g.get("call" + k + ".scaled").toFloatArray(), p);
                        int count = ((Number) horizons.get(k)).intValue() / p, from = in.patches() - count - 1;
                        long t0 = System.nanoTime();
                        float[] head = fwd.forward(in, from, count);
                        long ms = (System.nanoTime() - t0) / 1_000_000;
                        float[] expected = slice(g.get("call" + k + ".head"), from, count);
                        boolean[] keep = new boolean[in.rows() * count];
                        for (int r = 0; r < in.rows(); r++) {
                            for (int s = 0; s < count; s++) {
                                keep[r * count + s] = in.patchValid()[r * in.patches() + from + s];
                            }
                        }
                        double e = T0Golden.relativeMaxError(head, expected, T0Golden.widen(keep, p * q));
                        if (!(e <= limit)) {
                            failures.add(testCase + " call" + k + " head: " + e);
                        }
                        System.out.printf("%s call%d rows=%d patches=%d head %.2e  %d ms%n", testCase, k, in.rows(), in.patches(), e, ms);
                    }
                    int horizon = ((Number) info.get("H")).intValue();
                    T0Pipeline pipeline = new T0Pipeline(c, fwd, T0Golden.options(info), 1 << 20, Long.MAX_VALUE);
                    long t0 = System.nanoTime();
                    List<Forecast> out = pipeline.forecast(T0Golden.series(g, info), horizon);
                    long ms = (System.nanoTime() - t0) / 1_000_000;
                    double fe = forecastError(out, g.get("quantiles"));
                    if (!(fe <= limit)) {
                        failures.add(testCase + " forecast: " + fe);
                    }
                    System.out.printf("%s forecast %.2e  %d ms%n", testCase, fe, ms);
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    @Test
    public void mixedBatchMatchesTheCpuReference() throws IOException {
        try (T0Weights weights = T0Weights.load(T0Golden.checkpoint());
                TornadoT0Forward fwd = new TornadoT0Forward(weights.config(), T0Packed.pack(weights), 4)) {
            T0Config c = weights.config();
            double limit = fwd.gemmMode().reducedPrecision() ? REDUCED : TOLERANCE;
            List<TimeSeries> all = new ArrayList<>();
            int horizon = 48;
            for (String testCase : T0Golden.cases()) {
                Map<String, Object> info = T0Golden.caseInfo(testCase);
                try (SafeTensors g = T0Golden.open(testCase)) {
                    for (TimeSeries ts : T0Golden.series(g, info)) {
                        all.add(cutFuture(ts, horizon));
                    }
                }
            }
            T0Pipeline.Options o = T0Pipeline.Options.defaults(c).withRecommendedAdjustments();
            List<Forecast> dev = new T0Pipeline(c, fwd, o, 64, 8192).forecast(all, horizon);
            List<Forecast> ref = new T0Pipeline(c, new CpuT0(weights), o, 64, 8192).forecast(all, horizon);
            double worst = 0;
            for (int i = 0; i < all.size(); i++) {
                for (int t = 0; t < ref.get(i).targets(); t++) {
                    worst = Math.max(worst, T0Golden.relativeMaxError(flat(dev.get(i).values()[t]), flat(ref.get(i).values()[t]), null));
                }
            }
            System.out.printf("mixed batch of %d series: worst %.2e, shapes %s%n", all.size(), worst, fwd.cachedShapes());
            assertTrue("mixed batch: " + worst, worst <= limit);
        }
    }

    /** A series with its known-future covariates cut to the context plus {@code horizon}. */
    private static TimeSeries cutFuture(TimeSeries ts, int horizon) {
        if (ts.futureCount() == 0) {
            return ts;
        }
        float[][] f = new float[ts.futureCount()][];
        for (int i = 0; i < f.length; i++) {
            f[i] = java.util.Arrays.copyOf(ts.futureCovariates()[i], ts.length() + horizon);
        }
        return new TimeSeries(ts.target(), ts.pastCovariates(), f);
    }

    private static float[] flat(float[][] x) {
        float[] out = new float[x.length * x[0].length];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, out, i * x[0].length, x[0].length);
        }
        return out;
    }

    /** The golden head {@code (V, P, patch, Q)} at patches {@code [from, from + count)}. */
    private static float[] slice(Tensor head, int from, int count) {
        int rows = head.dim(0), n = head.dim(1), per = (int) (head.elements() / ((long) rows * n));
        float[] all = head.toFloatArray(), out = new float[rows * count * per];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(all, (r * n + from) * per, out, r * count * per, count * per);
        }
        return out;
    }

    private static double forecastError(List<Forecast> fc, Tensor golden) {
        int b = golden.dim(0), h = golden.dim(1), q = golden.dim(2);
        float[] exp = golden.toFloatArray(), act = new float[exp.length];
        int r = 0;
        for (Forecast f : fc) {
            for (int t = 0; t < f.targets(); t++, r++) {
                for (int s = 0; s < h; s++) {
                    for (int k = 0; k < q; k++) {
                        act[(r * h + s) * q + k] = f.values()[t][k][s];
                    }
                }
            }
        }
        return r == b ? T0Golden.relativeMaxError(act, exp, null) : Double.POSITIVE_INFINITY;
    }
}
