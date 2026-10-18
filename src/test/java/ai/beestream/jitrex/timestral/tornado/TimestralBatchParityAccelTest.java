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
package ai.beestream.jitrex.timestral.tornado;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ai.beestream.jitrex.api.TimeSeries;
import ai.beestream.jitrex.timestral.BlockInput;
import ai.beestream.jitrex.timestral.TimestralGolden;
import ai.beestream.jitrex.timestral.TimestralPipeline;
import ai.beestream.jitrex.timestral.TimestralWeights;
import ai.beestream.jitrex.timestral.cpu.CpuTimestral;
import org.junit.Test;

/**
 * The TornadoVM {@code predict_block} against the CPU reference on a large mixed batch the golden
 * cases do not cover: 37 series of different lengths, with gaps and a late start, some with past or
 * known-future covariates, so the attention tiles span several series and a series spans tiles.
 */
public class TimestralBatchParityAccelTest {

    private static final double TOLERANCE = 1e-4;
    /**
     * With FP16 or TF32 GEMM inputs ({@code -Dtimestral.tornado.gemm=half|cublas-fp16|cublas-tf32}): a
     * bound for a smoke check, not parity. Measured on Timestral-5M with FP16 GEMMs, the long
     * multi-block horizons are within 4e-2 of the reference's range and the rest within 1e-2.
     */
    private static final double REDUCED_TOLERANCE = 5e-2;

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
            pipeline.forecast(batch(new Random(11), 37, horizon), horizon);
            List<String> failures = new ArrayList<>();
            double limit = fwd.gemmMode().reducedPrecision() ? REDUCED_TOLERANCE : TOLERANCE;
            for (int k = 0; k < calls.size(); k++) {
                BlockInput b = calls.get(k);
                double err = TimestralGolden.relativeMaxError(flat(fwd.predictBlock(b)), flat(cpu.predictBlock(b)));
                System.out.printf("mixed batch call%d rows=%d length=%d horizon patches=%d covariates=%b  rel max err %.2e%n", k,
                        b.rows(), b.length(), b.horizonPatches(), b.role() != null, err);
                if (!(err <= limit)) {
                    failures.add("call" + k + ": " + err);
                }
            }
            assertTrue(String.join("\n", failures), failures.isEmpty());
        }
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
