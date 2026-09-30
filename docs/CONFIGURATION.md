# Configuring jitrex

[← back to the README](../README.md)

## JVM properties (`-D…`)

| Property | Values (default first) | Effect |
|---|---|---|
| `tirex.tornado.device` | `<backend>:<device>` (TornadoVM's default device) | Device to run on, numbered as in `tornado --devices` |
| `tirex.tornado.gemm` | `tiled`, `half`, `cublas`, `cublas-tf32`, `cublas-fp16`, `cublas-auto`, `cutile` | GEMM implementation (see *GEMM modes*) |
| `tirex.tornado.fp16MinTokens` | `1024` | `cublas-auto`: GEMMs over at least this many rows use FP16, smaller ones FP32 |
| `tirex.tornado.fp16Kernel` | `gemmex-mixed`, `gemmex-h16`, `lt-h16` | cuBLAS call for an FP16 GEMM: GemmEx FP16→FP32, GemmEx all-FP16, or cuBLASLt all-FP16 (the last two with an FP32 epilogue kernel) |
| `tirex.tornado.slstmR` | `fp16-reg`, `fp16`, `fp32` | The sLSTM recurrent weights R. `fp16-reg`: FP16, with half of each thread's R values held in registers for the whole sequence where that is faster (up to ~64 rows; it needs 512-thread work-groups and falls back to `fp16` elsewhere). `fp16`: FP16, streamed every step. `fp32`: FP32, streamed. The state is FP32 in all three |
| `tirex.tornado.slstmRows` | `0` (auto) | Rows per sLSTM work-group. Auto picks the largest value that still gives 64 work-groups |
| `tirex.tornado.mlstm` | `auto`, `chunk`, `step` | mLSTM kernel: `chunk` (blocks of 16 query steps share each staged key/value block) or `step` (one work-group per time step). `auto` takes `chunk` unless its grid would have fewer than 64 work-groups (a single 74-patch series) |
| `tirex.tornado.profile` | `false`, `true` | Print TornadoVM's per-task kernel times after every execution (analysis only: it slows every call) |
| `tirex.tornado.fuse` | `true`, `false` | Fused epilogues: bias + SiLU after the GEMM, residual + RMSNorm, sLSTM head norm + residual + RMSNorm |
| `tirex.tornado.cudaGraph` | `true` for `cublas*` / `cutile`, `false` otherwise | Capture and replay each task graph as a CUDA graph |

TornadoVM properties used alongside: `tornado.device.memory` (device heap, e.g. `15GB`), `tornado.cuda.nvcc` (for `cutile`).

## GEMM modes

| Mode | Kernel | Precision | Backends |
|---|---|---|---|
| `tiled` | `TiRexKernels.gemm` (KernelContext) | FP32 | all |
| `half` | `TiRexKernels.gemmHalf` (KernelContext, register-blocked) | FP16 in, FP32 accumulation and output | all |
| `cublas` / `cublas-tf32` | cuBLAS SGEMM | FP32 / TF32 | CUDA |
| `cublas-fp16` | cuBLAS GemmEx on every GEMM | FP16 in, FP32 accumulation | CUDA |
| `cublas-auto` | FP16 GemmEx from `fp16MinTokens` rows, SGEMM below | mixed | CUDA |
| `cutile` | `TiRexKernels.gemmHalfTile` (CUDA Tile, tensor cores) | FP16 in, FP32 accumulation | CUDA with toolkit 13.3+ |

## Programmatic options
- `ForecastPipeline(config, forward, batchSize, ttaDiff, ttaSignFlip)`: series per forward (512); trend differencing and sign flip, defaulting to the checkpoint's values.
- `TornadoForward(config, weights, maxCachedPlans)`: the number of cached plans, one per batch shape (8).

## Recommendations

| Platform | Recommended flags | Why |
|---|---|---|
| **NVIDIA** (tested: RTX 5070 Ti) | `-Dtirex.tornado.gemm=cublas-auto` (FP16 sLSTM weights in registers and CUDA graphs are the defaults), plus the cuBLAS 13 shim on machines that also have CUDA 12 cuBLAS | Fastest at every batch size: 2.41 ms at batch 1, ~4,850 series/s at batch 512. Error 1.5e-4 / 4.7e-5 |
| NVIDIA, closest to FP32 | `-Dtirex.tornado.gemm=cublas -Dtirex.tornado.slstmR=fp32` | FP32 GEMMs, error 2.7e-6 / 1.4e-6; slower than `cublas-auto` at large batches, where the FP16 tensor-core GEMMs pay off |
| NVIDIA, no vendor libraries | `-Dtirex.tornado.gemm=cutile` (toolkit 13.3+), otherwise `half` | `cutile`: 3.1 ms at batch 1, ~1,630 series/s at batch 256. `half`: 7.5 ms, ~1,100 series/s |
| **Apple Silicon** (tested: M4 Pro), Metal or OpenCL | `-Dtirex.tornado.gemm=half`, on either backend | `half` is 1.9–2.5× faster than `tiled` from batch 8 up; Metal and OpenCL are within ~5% of each other; see [PERFORMANCE.md](PERFORMANCE.md#apple-silicon) |
| Other OpenCL devices | `-Dtirex.tornado.gemm=half` (or `tiled` for FP32) | Portable; not tuned |
| Long contexts (GIFT-Eval checkpoints, 285 patches) | the platform's flags (the chunkwise mLSTM is the default), plus `-Dtirex.tornado.cudaGraph=false`, and batches of at most 128 series on a 16 GB GPU | `cublas-auto` is 1.8–2.5× faster than NXAI there ([PERFORMANCE.md](PERFORMANCE.md#long-context-gift-eval-checkpoints)). Device memory grows with rows × patches, and CUDA graphs can fail after many plan rebuilds (a TornadoVM issue under investigation) |

## Hardware and backends

| Hardware | TornadoVM backend | GEMM modes | Status |
|---|---|---|---|
| NVIDIA GPU | CUDA | `tiled`, `half`, `cublas`, `cublas-tf32`, `cublas-fp16`, `cublas-auto`, `cutile` | **Validated and benchmarked** on an RTX 5070 Ti (sm_120, Blackwell), driver 580, CUDA 13.0 |
| Apple Silicon GPU | Metal | `tiled`, `half` | **Validated** on an M4 Pro |
| Apple Silicon GPU | OpenCL | `tiled`, `half` | **Validated** on an M4 Pro |
| NVIDIA GPU | OpenCL | `tiled`, `half` | Expected to work (portable kernels), not tested |
| Intel / AMD GPUs, OpenCL CPU runtimes | OpenCL | `tiled`, `half` | Expected to work, not tested |
| Any CPU | none (plain JVM) | — | `CpuTiRex2Engine`: the FP32 reference, for validation and not for speed |

Some kernels adapt to the device's maximum work-group size (for example 256 on OpenCL on macOS, 1024 on Metal and CUDA). The fused sLSTM norm kernel needs 512 threads and is skipped on smaller devices.

## Hybrid APIs and CUDA features

| API | Used for | How to enable | Requirements |
|---|---|---|---|
| cuBLAS: `cublasSgemm`, `cublasSgemmTF32` | FP32 / TF32 GEMMs | `-Dtirex.tornado.gemm=cublas` / `cublas-tf32` | CUDA backend; `tornado-cublas` in the SDK |
| cuBLAS: `cublasGemmEx` (FP16 in, FP32 or FP16 out) | FP16 tensor-core GEMMs | `cublas-fp16`, `cublas-auto` | cuBLAS 13 on Blackwell (see [INSTALL.md](INSTALL.md#nvidia-extras)) |
| cuBLASLt: `ltMatmulFP16` | FP16 GEMMs | `-Dtirex.tornado.fp16Kernel=lt-h16` | as above |
| CUDA Tile (`TileContext`) | FP16 GEMM on tensor cores, written in TornadoVM | `-Dtirex.tornado.gemm=cutile` | CUDA Toolkit 13.3+ (`tileiras`), compute capability 8.0+, driver R580+ |
| CUDA graphs | replaying each task graph | `-Dtirex.tornado.cudaGraph=true` (default with `cublas*`, `cutile`) | CUDA backend |
| cuDNN | — | not used | — |

## Model features

| Feature | Supported |
|---|---|
| Checkpoints | `NX-AI/TiRex-2`, `-fevbench` (context 2048 + future 320 = 74 patches); `-gifteval-zs`, `-gifteval-pretrain` (8192 + 928 = 285 patches, no qk-norm) |
| Inputs | univariate, multi-target, past covariates, future-known covariates, mixed batches, NaN gaps, binary series, contexts shorter or longer than the model context |
| Test-time augmentation | sign-flip TTA and trend differencing, defaulting to the checkpoint's settings |
| Precision | FP32 throughout; optional FP16 GEMM inputs (FP32 accumulation and residual stream); optional FP16 sLSTM recurrent weights |
