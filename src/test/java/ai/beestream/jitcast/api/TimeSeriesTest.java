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
package ai.beestream.jitcast.api;

import static org.junit.Assert.assertThrows;

import org.junit.Test;

/** Series that no engine can forecast are rejected when they are built. */
public class TimeSeriesTest {

    @Test
    public void rejectsASeriesWithoutTimeSteps() {
        assertThrows(IllegalArgumentException.class, () -> TimeSeries.univariate(new float[0]));
    }

    @Test
    public void rejectsCovariatesOfAnotherLength() {
        assertThrows(IllegalArgumentException.class,
                () -> new TimeSeries(new float[][] {{1, 2, 3}}, new float[][] {{1, 2}}, null));
    }
}
