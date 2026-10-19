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
package ai.beestream.jitcast.tirex2.tornado;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.HashSet;
import java.util.Set;

import ai.beestream.jitcast.tirex2.ModelBatch;
import org.junit.Test;

/** Row bucketing: sizes, and padding rows that leave the batch's structure alone. */
public class RowBucketTest {

    @Test
    public void bucketsArePowersOfTwo() {
        int[][] cases = {{1, 1}, {2, 2}, {3, 4}, {4, 4}, {5, 8}, {17, 32}, {64, 64}, {65, 128}, {1000, 1024}};
        for (int[] c : cases) {
            assertEquals("rows " + c[0], c[1], TornadoForward.rowBucket(c[0]));
        }
    }

    @Test
    public void paddingRowsAreSingleRowTargetGroups() {
        // a series with one target and two covariates (group 0), then a univariate series (group 1)
        float[][] x = {{1f, 2f}, {3f, 4f}, {5f, 6f}, {7f, 8f}, {9f, 10f}};
        float[] group = {0f, 0f, 0f, 1f, 2f};
        boolean[] target = {true, false, false, true, true};
        ModelBatch padded = TornadoForward.padRows(new ModelBatch(x, group, target), 8);
        assertEquals(8, padded.rows());
        for (int r = 0; r < 5; r++) {
            assertSame(x[r], padded.x()[r]);
            assertEquals(group[r], padded.group()[r], 0f);
            assertEquals(target[r], padded.target()[r]);
        }
        Set<Float> groups = new HashSet<>();
        for (int r = 0; r < 5; r++) {
            groups.add(group[r]);
        }
        for (int r = 5; r < 8; r++) {
            assertArrayEquals(x[0], padded.x()[r], 0f);
            assertTrue("padding rows are targets", padded.target()[r]);
            assertTrue("each padding row has a new group of its own", groups.add(padded.group()[r]));
        }
    }
}
