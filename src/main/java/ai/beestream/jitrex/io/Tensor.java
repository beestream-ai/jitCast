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
package ai.beestream.jitrex.io;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/** One named tensor of a {@link SafeTensors} file: a row-major view over the mapped bytes. */
public record Tensor(String name, SafeTensors.DType dtype, long[] shape, MemorySegment data) {

    public long elements() {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        return n;
    }

    public int rank() {
        return shape.length;
    }

    public int dim(int i) {
        return Math.toIntExact(shape[i < 0 ? shape.length + i : i]);
    }

    /** Fails unless this tensor is F32 with exactly {@code expected} as its shape. */
    public Tensor requireF32(long... expected) {
        if (dtype != SafeTensors.DType.F32 || !Arrays.equals(shape, expected)) {
            throw new IllegalArgumentException(name + ": expected F32" + Arrays.toString(expected)
                    + ", found " + dtype + Arrays.toString(shape));
        }
        return this;
    }

    public float getFloat(long index) {
        return data.getAtIndex(SafeTensors.F32, index);
    }

    /** Copies an F32 tensor to a Java array. */
    public float[] toFloatArray() {
        if (dtype != SafeTensors.DType.F32) {
            throw new IllegalStateException(name + " is " + dtype + ", not F32");
        }
        return data.toArray(SafeTensors.F32);
    }

    @Override
    public String toString() {
        return name + ":" + dtype + Arrays.toString(shape);
    }
}
