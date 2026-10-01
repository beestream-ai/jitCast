# Configuring jitrex

[← back to the README](../README.md)

## JVM properties (`-D…`)

| Property | Values (default first) | Effect |
|---|---|---|
| `tirex.tornado.device` | `<backend>:<device>` (TornadoVM's default device) | Device to run on, numbered as in `tornado --devices` |
| `tirex.tornado.gemm` | `tiled`, `half`, `cublas`, `cublas-tf32`, `cublas-fp16`, `cublas-auto`, `cutile` | GEMM implementation (see *GEMM modes*) |
| `tirex.tornado.fp16MinTokens` | `0` | `cublas-auto`: GEMMs over at least this many rows use FP16, smaller ones FP32. 0 is FP16 everywhere, the fastest at every size with CUDA 13 cuBLAS; a threshold only helps where CUDA 12 cuBLAS is loaded |
| `tirex.tornado.fp16Kernel` | `gemmex-mixed`, `gemmex-h16`, `lt-h16` | cuBLAS call for an FP16 GEMM: GemmEx FP16→FP32, GemmEx all-FP16, or cuBLASLt all-FP16 (the last two with an FP32 epilogue kernel) |
| `tirex.tornado.slstmR` | `auto`, `fp16-reg`, `fp16`, `fp32` | The sLSTM recurrent weights R. `fp16-reg`: FP16, with half of each thread's R values held in registers for the whole sequence where that is faster (up to ~64 rows; it needs 512-thread work-groups and falls back to `fp16` elsewhere). `fp16`: FP16, streamed every step. `fp32`: FP32, streamed. `auto`: `fp16-reg` where that kernel is known to fit (CUDA compute capability 12 and up, Apple GPUs on Metal or OpenCL), `fp16` elsewhere (see below). The state is FP32 in all of them |
| `tirex.tornado.slstmRows` | `0` (auto) | Rows per sLSTM work-group. Auto picks the largest value that still gives 64 work-groups |
| `tirex.tornado.mlstm` | `auto`, `chunk`, `step` | mLSTM kernel: `chunk` (blocks of 16 query steps share each staged key/value block) or `step` (one work-group per time step). `auto` takes `chunk` unless its grid would have fewer than 64 work-groups (a single 74-patch series) |
| `tirex.tornado.rowBuckets` | `off`, `pow2` | `pow2` pads each batch's row count to a power of two, so different batch sizes share plans (each new plan compiles its kernels). The padding rows are computed, which costs time on every call whose size is not a power of two, so use it only when many sizes each occur a few times |
| `tirex.tornado.profile` | `false`, `true` | Print TornadoVM's per-task kernel times after every execution (analysis only: it slows every call) |
| `tirex.tornado.fuse` | `true`, `false` | Fused epilogues: bias + SiLU after the GEMM, residual + RMSNorm, sLSTM head norm + residual + RMSNorm |
| `tirex.tornado.singleGraph` | `false`, `true` | Build each plan as one task graph instead of a chain (input, one per block, output) sharing device buffers. Needs TornadoVM 7.1.0 or newer, whose task graphs can exceed 256 tasks |
| `tirex.tornado.cudaGraph` | `true` for `cublas*` / `cutile`, `false` otherwise | Capture and replay each task graph as a CUDA graph |
| `tornado.metal.dispatch.spinWait` (TornadoVM) | `False` (set by jitrex unless given) | TornadoVM develop's Metal backend spin-waits up to 1 ms on each dispatch before it blocks. A forward pass runs dozens of kernels per task graph, so the waits outlast the spin and jitrex runs up to 2× slower on Metal; jitrex turns it off. Code that calls TornadoVM before creating a `TornadoForward` should call `TornadoForward.tornadoDefaults()` first |
| `tirex.tornado.ffnHalf` | `true` | FP16 GEMM modes (`cublas-auto`, `cublas-fp16`, `half`): keep the FFN's intermediates in FP16 only, which cuts a plan's memory by about 15%; `false` keeps them in FP32 |
| `tirex.tornado.memReport` | `false` | Print each plan's activation buffers and its memory per token when the plan is built |
| `tirex.maxTokens` | `131072` | The most tokens (rows × patches) one forward pass takes; a forecast call is split into passes by this and by the batch size, which bounds a plan's memory (about 4 GB at the default) whatever the series' covariate counts. Forecasts do not depend on the split. `TiRexBench` ignores it unless set |
| `tornado.cuda.host.pinning` (TornadoVM) | `True`; `False` in `TiRexServer` | Pinned host buffers for transfers. Applications that keep several plans alive and evict them often (such as `TiRexServer`) should turn it off: on TornadoVM's CUDA backend that churn can make device allocations fail, and it gave no speed-up on fev-bench |

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
| **NVIDIA** (tested: A100) | `-Dtirex.tornado.gemm=cublas-auto` (CUDA graphs are the default), plus the cuBLAS 13 shim on machines that also have CUDA 12 cuBLAS | The fastest configuration: cuBLAS FP16 tensor-core GEMMs |
| NVIDIA, closest to FP32 | `-Dtirex.tornado.gemm=cublas -Dtirex.tornado.slstmR=fp32` | FP32 GEMMs; slower than `cublas-auto` at large batches, where the FP16 tensor-core GEMMs pay off |
| NVIDIA, no vendor libraries | `-Dtirex.tornado.gemm=half`, or `cutile` (toolkit 13.3+; it does not run on the A100 with TornadoVM develop yet) | TornadoVM's own FP16 GEMM |
| **Apple Silicon** (tested: M4 Pro), Metal or OpenCL | `-Dtirex.tornado.gemm=half`, on either backend | `half` is faster than `tiled` from small batches up; Metal and OpenCL run the same kernels |
| NVIDIA with OpenCL | `-Dtirex.tornado.gemm=half` | Works, but use CUDA where you can: OpenCL `half` is slower than CUDA `half` and has no vendor libraries or CUDA graphs |
| Other OpenCL devices | `-Dtirex.tornado.gemm=half` (or `tiled` for FP32) | Portable; not tuned |
| Long contexts (GIFT-Eval checkpoints, 285 patches) | the platform's flags (the chunkwise mLSTM is the default), plus `-Dtirex.tornado.cudaGraph=false`, and batch sizes that fit the device | Device memory grows with rows × patches; `tirex.maxTokens` bounds it per forward pass. With several cached plans and CUDA graphs, closing one plan could crash the others on TornadoVM's CUDA backend (fixed by [beehive-lab/TornadoVM#1160](https://github.com/beehive-lab/TornadoVM/pull/1160)); with an older TornadoVM keep one cached plan or turn CUDA graphs off |

## Hardware and backends

| Hardware | TornadoVM backend | GEMM modes | Status |
|---|---|---|---|
| NVIDIA GPU | CUDA | `tiled`, `half`, `cublas`, `cublas-tf32`, `cublas-fp16`, `cublas-auto`, `cutile` | **Validated and benchmarked** on an A100 80GB (sm_80, aarch64), driver 610, CUDA 13.4, without `cutile` |
| Apple Silicon GPU | Metal | `tiled`, `half` | **Validated and benchmarked** on an M4 Pro |
| Apple Silicon GPU | OpenCL | `tiled`, `half` | **Validated and benchmarked** on an M4 Pro |
| NVIDIA GPU | OpenCL | `tiled`, `half` | **Validated and benchmarked** on an A100 (goldens, fev-bench mini 20/20) |
| Intel / AMD GPUs, OpenCL CPU runtimes | OpenCL | `tiled`, `half` | Expected to work, not tested |
| Any CPU | none (plain JVM) | — | `CpuTiRex2Engine`: the FP32 reference, for validation and not for speed |

The register-resident sLSTM kernel (`slstmR=fp16-reg`) needs 512 threads per work-group with up to 128 registers each, and whether the compiler fits it depends on the GPU: it does for CUDA compute capability 12 (RTX 5070 Ti) and Apple GPUs, not for an A100 (sm_80), where CUDA refuses to launch it (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`) and NVIDIA's OpenCL returns wrong results without an error. The default, `slstmR=auto`, therefore uses it only on CUDA devices of compute capability 12 and up and on Apple GPUs, and the streamed FP16 kernel elsewhere; OpenCL does not report the compute capability, so on other OpenCL devices it stays with the streamed kernel. Setting `slstmR` explicitly overrides this. `cutile` does not run on the A100 with TornadoVM develop at 85eb7bed2: TornadoVM fails to parse `TileContext.checkShape` ("Unable to build sketch for method: gemmHalfTile").

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
