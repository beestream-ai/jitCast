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
package ai.beestream.jitcast.chronos2.tornado;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.chronos2.Chronos2Golden;
import ai.beestream.jitcast.chronos2.Chronos2Pipeline;
import ai.beestream.jitcast.chronos2.Chronos2Weights;
import ai.beestream.jitcast.chronos2.cpu.CpuChronos2;
import org.junit.Test;

/**
 * The TornadoVM forward pass against the CPU reference on a large mixed batch the golden cases do
 * not cover: 25 series of different lengths, with gaps and a late start, some multivariate, some
 * with past-only or known covariates, over a horizon long enough to unroll, so the attention tiles
 * span several groups, a group spans tiles, and the unrolled groups are not contiguous rows.
 */
public class Chronos2BatchParityAccelTest {

    private static final double TOLERANCE = 1e-4;
    /** With FP16 or TF32 GEMM inputs ({@code -Dchronos2.tornado.gemm=half|cublas-fp16|cublas-tf32}): a smoke check, not parity. */
    private static final double REDUCED_TOLERANCE = 5e-2;

    @Test
    public void mixedBatchMatchesTheCpuReference() throws IOException {
        try (Chronos2Weights weights = Chronos2Weights.load(Chronos2Golden.checkpoint());
                Chronos2Forward fwd = new Chronos2Forward(weights.config(), Chronos2Packed.pack(weights))) {
            CpuChronos2 cpu = new CpuChronos2(weights);
            List<String> failures = new ArrayList<>();
            double limit = fwd.gemmMode().reducedPrecision() ? REDUCED_TOLERANCE : TOLERANCE;
            int[] call = {0};
            // four patches per pass, so a horizon of 100 unrolls once with short contexts
            Chronos2Pipeline.Options o = Chronos2Pipeline.Options.defaults(weights.config());
            o = new Chronos2Pipeline.Options(o.contextLength(), false, 4, o.unrolledQuantiles());
            Chronos2Pipeline pipeline = new Chronos2Pipeline(weights.config(), in -> {
                float[][][] out = fwd.forward(in);
                double err = Chronos2Golden.relativeMaxError(Chronos2Golden.flat(out), Chronos2Golden.flat(cpu.forward(in)));
                System.out.printf("mixed batch call%d rows=%d length=%d output patches=%d  rel max err %.2e%n", call[0], in.rows(),
                        in.length(), in.outputPatches(), err);
                if (!(err <= limit)) {
                    failures.add("call" + call[0] + ": " + err);
                }
                call[0]++;
                return out;
            }, o, 1000, Chronos2Pipeline.defaultMaxTokens());
            int horizon = 100;
            pipeline.forecast(batch(new Random(11), 25, horizon), horizon);
            assertTrue("the horizon unrolls", call[0] == 2);
            assertTrue(String.join("\n", failures), failures.isEmpty());
        }
    }

    static List<TimeSeries> batch(Random rnd, int count, int horizon) {
        List<TimeSeries> out = new ArrayList<>();
        for (int s = 0; s < count; s++) {
            int len = 40 + rnd.nextInt(260);
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
}
