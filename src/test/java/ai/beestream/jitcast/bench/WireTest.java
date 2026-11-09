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
package ai.beestream.jitcast.bench;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.junit.Test;

/** The harness servers check every size a client sends before allocating anything. */
public class WireTest {

    private static Wire wire(int... ints) {
        ByteBuffer bb = ByteBuffer.allocate(4 * ints.length).order(ByteOrder.LITTLE_ENDIAN);
        for (int i : ints) {
            bb.putInt(i);
        }
        return new Wire(new DataInputStream(new ByteArrayInputStream(bb.array())));
    }

    @Test
    public void readsCountsInRange() throws Exception {
        assertEquals(3, wire(3).count("series count", 1, 10));
    }

    @Test
    public void rejectsCountsOutOfRange() {
        assertThrows(Wire.BadRequest.class, () -> wire(-1).count("series count", 1, 10));
        assertThrows(Wire.BadRequest.class, () -> wire(11).count("series count", 1, 10));
    }

    @Test
    public void rejectsRequestsLargerThanTheBudget() {
        Wire w = wire();
        // 2^16 × 2^16 values would overflow an int byte count and exceed the budget
        assertThrows(Wire.BadRequest.class, () -> w.floats(1 << 16, 1 << 16));
    }
}
