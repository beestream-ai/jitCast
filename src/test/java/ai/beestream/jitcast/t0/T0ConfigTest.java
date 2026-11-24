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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Test;

import ai.beestream.jitcast.api.TimeSeries;

/** The configuration parser, the native batch layout and the quantile-level plans, without a checkpoint. */
public class T0ConfigTest {

    private static final String GAMMA = """
            {"dropout": 0.1, "dtype": null, "embed_dim": 1024, "group_every_n": 3, "mlp_hidden_dim": 2048,
             "num_heads": 8, "num_layers": 24, "patch_size": 32,
             "quantile_levels": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
             "scaler_eps": 0.01, "scaler_eps_mode": "std_clamp", "scaler_use_arcsinh": true,
             "scaler_zero_no_value": false, "xpos_scale_layout": "halves"}""";

    @Test
    public void parsesTheGammaConfig() {
        T0Config c = T0Config.fromJson(GAMMA);
        assertEquals(128, c.headDim());
        assertEquals(21, c.quantiles());
        assertFalse(c.scalerEpsVarianceOffset());
        assertTrue(c.xposHalves());
        assertFalse(c.scalerZeroNoValue());
        assertFalse(c.isGroupLayer(0));
        assertTrue(c.isGroupLayer(2));
        assertTrue(c.isGroupLayer(23));
    }

    @Test
    public void missingFieldsTakeTheAlphaDefaults() {
        T0Config c = T0Config.fromJson("""
                {"dropout": 0.1, "embed_dim": 512, "group_every_n": 3, "mlp_hidden_dim": 2048, "num_heads": 8,
                 "num_layers": 24, "patch_size": 32, "quantile_levels": [0.1, 0.25, 0.5, 0.75, 0.9],
                 "scaler_use_arcsinh": true}""");
        assertEquals(0.1f, c.scalerEps(), 0f);
        assertTrue(c.scalerEpsVarianceOffset());
        assertFalse(c.xposHalves());
        assertTrue(c.scalerZeroNoValue());
    }

    @Test
    public void rejectsWhatItDoesNotImplement() {
        assertThrows(IllegalArgumentException.class, () -> T0Config.fromJson(GAMMA.replace("\"dropout\"", "\"extra\": 1, \"dropout\"")));
        assertThrows(IllegalArgumentException.class, () -> T0Config.fromJson(GAMMA.replace("std_clamp", "other")));
        assertThrows(IllegalArgumentException.class, () -> T0Config.fromJson(GAMMA.replace("\"halves\"", "\"rows\"")));
        assertThrows(IllegalArgumentException.class, () -> T0Config.fromJson(GAMMA.replace("\"dtype\": null", "\"dtype\": \"bfloat16\"")));
        assertThrows(IllegalArgumentException.class, () -> T0Config.fromJson(GAMMA.replace("\"num_heads\": 8", "\"num_heads\": 7")));
    }

    @Test
    public void batchesSeriesRightAligned() {
        TimeSeries a = new TimeSeries(new float[][] {{1, Float.NaN, 3}}, new float[][] {{4, 5, 6}},
                new float[][] {{7, 8, 9, 10, 11}});
        TimeSeries b = TimeSeries.univariate(new float[] {1, 2});
        T0Batch batch = T0Batch.of(List.of(a, b), 2);
        assertEquals(4, batch.rows());
        assertEquals(5, batch.width());
        assertEquals(3, batch.contextEnd());
        assertArrayEquals(new int[] {0, 3}, batch.targetRows());
        assertArrayEquals(new byte[] {0, 2, 0, 4, 4}, java.util.Arrays.copyOfRange(batch.mask(), 0, 5));
        assertEquals(T0Batch.HISTORICAL, batch.rowType(1));
        assertEquals(T0Batch.FUTURE, batch.rowType(2));
        assertArrayEquals(new byte[] {1, 0, 0, 4, 4}, java.util.Arrays.copyOfRange(batch.mask(), 15, 20));
        assertEquals(-1, batch.group()[15]);
        assertEquals(1, batch.rowGroup(3));
    }

    @Test
    public void rolloutLevelsKeepTheTailKnots() {
        double[] trained = T0Config.fromJson(GAMMA).quantileLevels();
        assertArrayEquals(new double[] {0.1, 0.5, 0.9}, T0Quantiles.rolloutLevels(trained, new double[] {0.1, 0.5, 0.9}), 0);
        assertArrayEquals(new double[] {0.01, 0.05, 0.5, 0.95, 0.99},
                T0Quantiles.rolloutLevels(trained, new double[] {0.001, 0.5, 0.999}), 0);
        float[] tails = T0Quantiles.extrapolate(new double[] {0.001, 0.5, 0.999}, new double[] {0.01, 0.05, 0.5, 0.95, 0.99},
                new float[] {-2, -1, 0, 1, 2}, trained);
        assertTrue(tails[0] < -2 && tails[2] > 2);
        assertEquals(0f, tails[1], 0f);
    }
}
