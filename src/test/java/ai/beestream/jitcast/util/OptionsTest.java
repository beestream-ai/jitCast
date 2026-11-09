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
package ai.beestream.jitcast.util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.After;
import org.junit.Test;

/** Options fail on values that do not parse instead of falling back to their defaults. */
public class OptionsTest {

    private static final String P = "jitcast.test.option";

    private enum Mode { AUTO, CUBLAS_FP16 }

    @After
    public void clear() {
        System.clearProperty(P);
    }

    @Test
    public void defaultsWhenUnset() {
        assertEquals(7, Options.integer(P, 7, 1));
        assertEquals(Mode.AUTO, Options.enumeration(P, Mode.AUTO, Mode.class));
        assertTrue(Options.bool(P, true));
    }

    @Test
    public void parsesValidValues() {
        System.setProperty(P, "8GB");
        assertEquals(8L << 30, Options.bytes(P, 1, 1));
        System.setProperty(P, "cublas-fp16");
        assertEquals(Mode.CUBLAS_FP16, Options.enumeration(P, Mode.AUTO, Mode.class));
        System.setProperty(P, "FALSE");
        assertEquals(false, Options.bool(P, true));
    }

    @Test
    public void rejectsBadValues() {
        System.setProperty(P, "8GB");
        assertThrows(IllegalArgumentException.class, () -> Options.integer(P, 1, 1));
        System.setProperty(P, "yes");
        assertThrows(IllegalArgumentException.class, () -> Options.bool(P, false));
        System.setProperty(P, "fp16x");
        assertThrows(IllegalArgumentException.class, () -> Options.choice(P, "auto", "auto", "fp16"));
        System.setProperty(P, "0");
        assertThrows(IllegalArgumentException.class, () -> Options.integer(P, 1, 1));
        System.setProperty(P, "12 parsecs");
        assertThrows(IllegalArgumentException.class, () -> Options.bytes(P, 1, 1));
    }
}
