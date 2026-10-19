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
package ai.beestream.jitcast.tornado;

import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;

/**
 * How a family's GEMMs run.
 *
 * <ul>
 *   <li>{@code TILED}: {@link Kernels#gemm}, FP32, any backend.
 *   <li>{@code BLOCKED}: {@link Kernels#gemmBlocked} from a minimum row count and {@link Kernels#gemm}
 *       below it, FP32, any backend.
 *   <li>{@code HALF}: {@link Kernels#gemmHalf}, FP16 inputs with FP32 accumulation and output, any
 *       backend.
 *   <li>{@code CUBLAS}, {@code CUBLAS_TF32}: cuBLAS SGEMM in FP32 or TF32 (CUDA).
 *   <li>{@code CUBLAS_FP16}: cuBLAS GemmEx with FP16 inputs, FP32 accumulation and output, on tensor
 *       cores (CUDA).
 *   <li>{@code AUTO}: {@code CUBLAS} on a CUDA device, {@code BLOCKED} elsewhere.
 * </ul>
 */
public enum GemmMode {
    AUTO, TILED, BLOCKED, HALF, CUBLAS, CUBLAS_TF32, CUBLAS_FP16;

    /** Parses {@code tiled}, {@code cublas-fp16} and so on. */
    public static GemmMode parse(String v) {
        return valueOf(v.trim().toUpperCase().replace('-', '_'));
    }

    /** {@code AUTO} made concrete for a device. */
    public GemmMode resolve(TornadoDevice device) {
        if (this != AUTO) {
            return this;
        }
        return device.getTornadoVMBackend() == TornadoVMBackendType.CUDA ? CUBLAS : BLOCKED;
    }

    /** Whether the GEMMs run in cuBLAS, which needs the CUDA backend. */
    public boolean cublas() {
        return this == CUBLAS || this == CUBLAS_TF32 || this == CUBLAS_FP16;
    }

    /** Whether the GEMM inputs are rounded below FP32 (FP16 or TF32). */
    public boolean reducedPrecision() {
        return this == HALF || this == CUBLAS_TF32 || this == CUBLAS_FP16;
    }
}
