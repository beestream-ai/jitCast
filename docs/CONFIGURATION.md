# Configuring jitcast

[← back to the README](../README.md)

Engine options are JVM properties (`-D…`), read when an engine is created. A value that does not
parse, or a GEMM mode the device cannot run, fails the engine's creation with the property's name;
nothing falls back to a default silently. The options every model shares come first; each model then
has its own, with its own prefix (`tirex.*`, `timestral.*`, `chronos2.*`).

An engine can be shared between threads: a TornadoVM engine runs their calls one at a time, since
every call of a shape reuses that shape's device buffers.

## Shared options

| Property | Values (default first) | Effect |
|---|---|---|
| `jitcast.tornado.device` | `<backend>:<device>` (TornadoVM's default device) | Device to run on, numbered as in `tornado --devices` |
| `tornado.metal.dispatch.spinWait` (TornadoVM) | `False` (set by jitcast unless given) | TornadoVM's Metal backend spin-waits up to 1 ms on each dispatch before it blocks. A forward pass runs dozens of kernels per task graph, so the waits outlast the spin and jitcast runs up to 2× slower on Metal; jitcast turns it off, for every model. Code that calls TornadoVM before creating an engine should call `tornado.Devices.defaults()` first |
| `tornado.cuda.host.pinning` (TornadoVM) | `True`; `False` in `TiRexServer` and `TimestralServer` | Pinned host buffers for transfers. Applications that keep several plans alive and evict them often (such as the harness servers) should turn it off: on TornadoVM's CUDA backend that churn can make device allocations fail, and it gave no speed-up on fev-bench |

TornadoVM properties used alongside: `tornado.device.memory` (TornadoVM's cap on the device memory it
allocates, 4 GB unless set, e.g. `15GB`), `tornado.cuda.nvcc` (for CUDA Tile).

### Compiled plans

Each model compiles one plan per input shape and keeps the most recently used ones. TornadoVM gives
every plan its own device buffers, a copy of the model's weights included, so a plan's footprint is
its weights plus its activations. A cache holds at most `<model>.tornado.plans` plans (8) and keeps
their device memory, as TornadoVM reports it after each plan's first run, within
`<model>.tornado.planBytes` (by default `tornado.device.memory`), evicting the least recently used
first; a plan larger than the budget still runs, alone. A plan whose first run fails is not kept;
if other plans were cached, they are evicted and the plan is built once more, in case the device
was out of memory.

## Hardware and backends

| Hardware | TornadoVM backend | TiRex-2 | Timestral | Chronos-2 |
|---|---|---|---|---|
| NVIDIA GPU | CUDA | validated and benchmarked (A100, A10) | validated and benchmarked (A100, RTX 5070 Ti, A10) | validated and benchmarked (RTX 5070 Ti, A10) |
| NVIDIA GPU | OpenCL | validated (A100) | validated (A100) | not tested |
| Apple Silicon GPU | Metal | validated (M4 Pro) | validated and benchmarked (M4 Pro) | validated and benchmarked (M4 Pro) |
| Apple Silicon GPU | OpenCL | validated (M4 Pro) | validated and benchmarked (M4 Pro) | validated (M4 Pro) |
| Intel / AMD GPUs, OpenCL CPU runtimes | OpenCL | expected to work, not tested | expected to work, not tested | expected to work, not tested |
| Any CPU | none (plain JVM) | `CpuTiRex2Engine`, the FP32 reference | `CpuTimestralEngine`, the FP32 reference | `CpuChronos2Engine`, the FP32 reference |

The CPU engines are the references the device kernels are validated against, not fast paths.

## Hybrid APIs and CUDA features

| API | Used for | How to enable | Requirements |
|---|---|---|---|
| cuBLAS: `cublasSgemm`, `cublasSgemmTF32` | FP32 / TF32 GEMMs | `tirex.tornado.gemm` / `timestral.tornado.gemm` / `chronos2.tornado.gemm` = `cublas`, `cublas-tf32` | CUDA backend; `tornado-cublas` in the SDK |
| cuBLAS: `cublasGemmEx` (FP16 in, FP32 or FP16 out) | FP16 tensor-core GEMMs | `cublas-fp16` (all), `cublas-auto` (TiRex-2) | cuBLAS 13 on Blackwell (see [INSTALL.md](INSTALL.md#nvidia-extras)) |
| cuBLASLt: `ltMatmulFP16` | FP16 GEMMs | `-Dtirex.tornado.fp16Kernel=lt-h16` (TiRex-2) | as above |
| CUDA Tile (`TileContext`) | FP16 GEMM on tensor cores, written in TornadoVM | `-Dtirex.tornado.gemm=cutile` (TiRex-2) | TornadoVM 8, CUDA Toolkit 13.3+ (`tileiras`), compute capability 8.0+, driver R580+ |
| CUDA graphs | replaying the plan's task graph | `tirex.tornado.cudaGraph`, `timestral.tornado.cudaGraph`, `chronos2.tornado.cudaGraph` (on by default with cuBLAS) | CUDA backend |
| cuDNN | — | not used | — |

## TiRex-2

| Property | Values (default first) | Effect |
|---|---|---|
| `tirex.tornado.gemm` | `auto`, `tiled`, `half`, `cublas`, `cublas-tf32`, `cublas-fp16`, `cublas-auto`, `cutile` | GEMM implementation (see *GEMM modes*). `auto`: `cublas` (FP32) on CUDA, `tiled` elsewhere |
| `tirex.tornado.plans` | `8` | Shapes kept compiled (rows, known covariates, rows in multi-row groups); see *Compiled plans* |
| `tirex.tornado.planBytes` | `tornado.device.memory` | The device memory the cached plans may hold together (bytes, or `16GB` and so on) |
| `tirex.tornado.fp16MinTokens` | `0` | `cublas-auto`: GEMMs over at least this many rows use FP16, smaller ones FP32. 0 is FP16 everywhere, the fastest at every size with CUDA 13 cuBLAS; a threshold only helps where CUDA 12 cuBLAS is loaded |
| `tirex.tornado.fp16Kernel` | `gemmex-mixed`, `gemmex-h16`, `lt-h16` | cuBLAS call for an FP16 GEMM: GemmEx FP16→FP32, GemmEx all-FP16, or cuBLASLt all-FP16 (the last two with an FP32 epilogue kernel) |
| `tirex.tornado.slstmR` | `auto`, `fp16-reg`, `fp16`, `fp32` | The sLSTM recurrent weights R. `fp16-reg`: FP16, with half of each thread's R values held in registers for the whole sequence where that is faster (up to ~64 rows; it needs 512-thread work-groups and falls back to `fp16` elsewhere). `fp16`: FP16, streamed every step. `fp32`: FP32, streamed. `auto`: `fp16-reg` where that kernel is known to fit (CUDA compute capability 12 and up, Apple GPUs on Metal or OpenCL), `fp16` elsewhere (see below). The state is FP32 in all of them |
| `tirex.tornado.slstmRows` | `0` (auto) | Rows per sLSTM work-group. Auto picks the largest value that still gives 64 work-groups |
| `tirex.tornado.mlstm` | `auto`, `chunk`, `step` | mLSTM kernel: `chunk` (blocks of 16 query steps share each staged key/value block) or `step` (one work-group per time step). `auto` takes `chunk` unless its grid would have fewer than 64 work-groups (a single 74-patch series) |
| `tirex.tornado.rowBuckets` | `off`, `pow2` | `pow2` pads each batch's row count to a power of two, so different batch sizes share plans (each new plan compiles its kernels). The padding rows are computed, which costs time on every call whose size is not a power of two, so use it only when many sizes each occur a few times |
| `tirex.tornado.profile` | `false`, `true` | Print TornadoVM's per-task kernel times after every execution (analysis only: it slows every call) |
| `tirex.tornado.fuse` | `true`, `false` | Fused epilogues: bias + SiLU after the GEMM, residual + RMSNorm, sLSTM head norm + residual + RMSNorm |
| `tirex.tornado.cudaGraph` | `true` for `cublas*` / `cutile`, `false` otherwise | Capture and replay the plan's task graph as a CUDA graph |
| `tirex.tornado.ffnHalf` | `true` | FP16 GEMM modes (`cublas-auto`, `cublas-fp16`, `half`): keep the FFN's intermediates in FP16 only, which cuts a plan's memory by about 15%; `false` keeps them in FP32 |
| `tirex.tornado.memReport` | `false` | Print each plan's activation buffers and its memory per token when the plan is built |
| `tirex.maxTokens` | `131072` | The most tokens (rows × patches) one forward pass takes; a forecast call is split into passes by this and by the batch size, which bounds a plan's memory (about 4 GB at the default) whatever the series' covariate counts. Forecasts do not depend on the split. `TiRexBench` ignores it unless set |

### GEMM modes

| Mode | Kernel | Precision | Backends |
|---|---|---|---|
| `tiled` | `Kernels.gemm` (KernelContext) | FP32 | all |
| `half` | `Kernels.gemmHalf` (KernelContext, register-blocked) | FP16 in, FP32 accumulation and output | all |
| `cublas` / `cublas-tf32` | cuBLAS SGEMM | FP32 / TF32 | CUDA |
| `cublas-fp16` | cuBLAS GemmEx on every GEMM | FP16 in, FP32 accumulation | CUDA |
| `cublas-auto` | FP16 GemmEx from `fp16MinTokens` rows, SGEMM below | mixed | CUDA |
| `cutile` | `Kernels.gemmHalfTile` (CUDA Tile, tensor cores) | FP16 in, FP32 accumulation | CUDA with toolkit 13.3+ |

These are TiRex-2's modes; Timestral's are [below](#timestral).

### Programmatic options

- `ForecastPipeline(config, forward, batchSize, ttaDiff, ttaSignFlip)`: series per forward (512); trend differencing and sign flip, defaulting to the checkpoint's values.
- `TornadoForward(config, weights, maxCachedPlans)`: the number of cached plans, one per batch shape (8).

### Recommendations

| Platform | Recommended flags | Why |
|---|---|---|
| **NVIDIA** (tested: A100, A10) | `-Dtirex.tornado.gemm=cublas-auto` (CUDA graphs are the default), plus, with TornadoVM 7.3.1 or earlier, the cuBLAS 13 shim on machines that also have CUDA 12 cuBLAS | The fastest configuration: cuBLAS FP16 tensor-core GEMMs |
| NVIDIA, closest to FP32 | `-Dtirex.tornado.gemm=cublas -Dtirex.tornado.slstmR=fp32` | FP32 GEMMs; slower than `cublas-auto` at large batches, where the FP16 tensor-core GEMMs pay off |
| NVIDIA, no vendor libraries | `-Dtirex.tornado.gemm=cutile` (TornadoVM 8, toolkit 13.3+), or `half` | TornadoVM's own FP16 GEMMs; `cutile` on tensor cores is 2.3× faster than `half` at one series on an A10 |
| **Apple Silicon** (tested: M4 Pro), Metal or OpenCL | `-Dtirex.tornado.gemm=half`, on either backend | `half` is faster than `tiled` from small batches up; Metal and OpenCL run the same kernels |
| NVIDIA with OpenCL | `-Dtirex.tornado.gemm=half` | Works, but use CUDA where you can: OpenCL `half` is slower than CUDA `half` and has no vendor libraries or CUDA graphs |
| Other OpenCL devices | `-Dtirex.tornado.gemm=half` (or `tiled` for FP32) | Portable; not tuned |
| Long contexts (GIFT-Eval checkpoints, 285 patches) | the platform's flags (the chunkwise mLSTM is the default), plus `-Dtirex.tornado.cudaGraph=false`, and batch sizes that fit the device | Device memory grows with rows × patches; `tirex.maxTokens` bounds it per forward pass. With several cached plans and CUDA graphs, closing one plan could crash the others on TornadoVM's CUDA backend (fixed by [beehive-lab/TornadoVM#1160](https://github.com/beehive-lab/TornadoVM/pull/1160)); with an older TornadoVM keep one cached plan or turn CUDA graphs off |

### Kernel choices by device

The register-resident sLSTM kernel (`slstmR=fp16-reg`) needs 512 threads per work-group with up to 128 registers each, and whether the compiler fits it depends on the GPU: it does for CUDA compute capability 12 and Apple GPUs, not for an A100 (sm_80), where CUDA refuses to launch it (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`) and NVIDIA's OpenCL returns wrong results without an error. The default, `slstmR=auto`, therefore uses it only on CUDA devices of compute capability 12 and up and on Apple GPUs, and the streamed FP16 kernel elsewhere; OpenCL does not report the compute capability, so on other OpenCL devices it stays with the streamed kernel. Setting `slstmR` explicitly overrides this. `cutile` needs TornadoVM 8: releases up to 7.3.1 fail to parse `TileContext.checkShape` ("Unable to build sketch for method: gemmHalfTile").

Some kernels adapt to the device's maximum work-group size (for example 256 on OpenCL on macOS, 1024 on Metal and CUDA). The fused sLSTM norm kernel needs 512 threads and is skipped on smaller devices.

## Timestral

| Property | Values (default first) | Effect |
|---|---|---|
| `timestral.tornado.gemm` | `auto`, `simdgroup`, `blocked`, `tiled`, `half`, `cublas`, `cublas-tf32`, `cublas-fp16` | GEMM implementation. `auto`: `cublas` on CUDA, `simdgroup` on Metal when Metal is TornadoVM's default backend (`-Dtornado.metal.priority=100`), `blocked` elsewhere. `simdgroup`: FP32 on Apple's simdgroup matrix units (TornadoVM 7.3.1+). `blocked`: TornadoVM's register-blocked FP32 kernel from `blockedMinRows` rows and the tiled one under it. `half`: TornadoVM's FP16 kernel (FP32 accumulation). `cublas*`: cuBLAS in FP32, TF32, or FP16 with FP32 accumulation |
| `timestral.tornado.blockedMinRows` | `128` | The fewest GEMM rows that use the register-blocked kernel (its 64-row tiles waste most of their work below that) |
| `timestral.tornado.wideMinRows` | `1024` | The fewest GEMM rows that use the register-blocked kernel with 64 × 128 tiles (4 × 8 outputs per thread), which leave the GPU idle with fewer rows |
| `timestral.tornado.simdgroupWideMinRows` | `64` | The fewest GEMM rows that use the 64 × 64 simdgroup kernel; fewer use the 32 × 32 one, which spreads over more work-groups |
| `timestral.tornado.splitKTarget` | `320` | Split K: a 64 × 64 simdgroup GEMM with fewer tiles than this also splits K across work-groups (into slices of at least 128) and adds the partial tiles in a second kernel, so GEMMs with few rows fill the GPU. `1` turns it off |
| `timestral.tornado.splitKMax` | `16` | The most K slices of a split GEMM |
| `timestral.tornado.cudaGraph` | `true` with cuBLAS, `false` otherwise | Capture the plan's task graph once and replay it as a CUDA graph |
| `timestral.tornado.attention` | `auto`, `kernel`, `cublas` | Time attention. `auto`: with a cuBLAS GEMM mode, as cuBLAS FP32 batched GEMMs (scores, then the masked softmax, then the weighted values) for blocks of at least `blasAttentionMinTokens` tokens per row and at most `blasAttentionMaxRows` rows, and TornadoVM's attention kernel otherwise and on other backends (the batched GEMMs compute every score, masked or not, and lose to the kernel on large batches). `cublas`: always with a cuBLAS GEMM mode; `kernel`: never |
| `timestral.tornado.blasAttentionMinTokens` | `128` | The fewest tokens per row (patches of context plus horizon) that use the cuBLAS time attention under `auto` |
| `timestral.tornado.blasAttentionMaxRows` | `8` | The most rows (series × variates) a block may have to use the cuBLAS time attention under `auto` |
| `timestral.tornado.attentionScoreBytes` | `536870912` (512 MB) | The most memory the cuBLAS time attention's scores take at once; larger blocks run in chunks of (row, head) batches |
| `timestral.tornado.buckets` | `true`, `false` | Pad each block to a shared shape (the context with leading empty patches, the rows with empty series, each by at most a quarter), so that few plans are compiled; the forecasts do not change |
| `timestral.tornado.plans` | `8` | Block shapes kept compiled (rows, encoder tokens, horizon patches, covariates); see *Compiled plans* |
| `timestral.tornado.planBytes` | `tornado.device.memory` | The device memory the cached plans may hold together (bytes, or `16GB` and so on) |
| `timestral.tornado.profile` | `false`, `true` | Print TornadoVM's per-task kernel times after every execution (analysis only) |
| `timestral.timing` | `false`, `true` | Print the time of each host and device phase per forecast (analysis only) |
| `timestral.maxTokens` | `131072` | The most tokens (rows × patches) one forward pass takes; forecasts do not depend on the split |

`TimestralPipeline.Options` sets the forecaster: the context cap, the patches decoded per block, the
cap to the observed range, the threshold for fed-back medians, sign-flip averaging and non-negative
clipping, with the reference's defaults.

### Recommendations

| Platform | Recommended flags | Why |
|---|---|---|
| **NVIDIA** (tested: A100, A10, RTX 5070 Ti) | the defaults (`auto`: cuBLAS FP32 with CUDA graphs) | Matches the reference to 1e-5 and is faster than it from 1 to 512 series on every checkpoint tried |
| NVIDIA, fastest | `-Dtimestral.tornado.gemm=cublas-fp16` | 3–4× faster again on Timestral-209M from 32 series up; forecasts move by up to 1e-2 of the series' range (see [VALIDATION.md](https://github.com/beestream-ai/jitCast-benchmarking/blob/main/docs/VALIDATION.md#timestral)) |
| **Apple Silicon** (tested: M4 Pro) | the defaults on Metal (`auto`: `simdgroup`; with an SDK that also has OpenCL, add `-Dtornado.metal.priority=100`) | The simdgroup GEMMs are 2–4× faster than the portable `blocked` ones on the larger checkpoints; Metal is faster than OpenCL on Apple GPUs |
| Other OpenCL devices | the defaults | Portable; not tuned |

## Chronos-2

| Property | Values (default first) | Effect |
|---|---|---|
| `chronos2.tornado.gemm` | `auto`, `simdgroup`, `blocked`, `tiled`, `half`, `cublas`, `cublas-tf32`, `cublas-fp16` | GEMM implementation, as `timestral.tornado.gemm`: `cublas` on CUDA, `simdgroup` on Metal when Metal is TornadoVM's default backend, `blocked` elsewhere |
| `chronos2.tornado.blockedMinRows` | `128` | The fewest GEMM rows that use the register-blocked kernel |
| `chronos2.tornado.wideMinRows` | `1024` | The fewest GEMM rows that use the register-blocked kernel with 64 × 128 tiles |
| `chronos2.tornado.simdgroupWideMinRows` | `64` | The fewest GEMM rows that use the 64 × 64 simdgroup kernel |
| `chronos2.tornado.splitKTarget` | `320` | Split K for 64 × 64 simdgroup GEMMs with fewer tiles than this (`1`: off) |
| `chronos2.tornado.splitKMax` | `16` | The most K slices of a split GEMM |
| `chronos2.tornado.cudaGraph` | `true` with cuBLAS, `false` otherwise | Capture the plan's task graph once and replay it as a CUDA graph |
| `chronos2.tornado.plans` | `8` | Shapes kept compiled (rows, context patches, output patches); see *Compiled plans* |
| `chronos2.tornado.planBytes` | `tornado.device.memory` | The device memory the cached plans may hold together (bytes, or `16GB` and so on) |
| `chronos2.tornado.profile` | `false`, `true` | Print TornadoVM's per-task kernel times after every execution (analysis only) |
| `chronos2.maxTokens` | `131072` | The most tokens (rows × tokens per row) one forward pass takes; larger passes run as several, split between groups, and the forecasts do not depend on the split. A group is never split, so with cross-learning (one group per batch) a batch can exceed it |

`Chronos2Pipeline.Options` sets the pipeline: the context cap, cross-learning, the patches per forward
pass before the long-horizon unrolling and the unrolled quantile levels, with the reference's
defaults; `withBatchSize` sets the rows per batch (256).

### Recommendations

| Platform | Recommended flags | Why |
|---|---|---|
| **NVIDIA** (tested: RTX 5070 Ti, A10) | the defaults (`auto`: cuBLAS FP32 with CUDA graphs) | Matches the reference to 4.5e-5 and is level with it or faster from 1 to 512 series |
| NVIDIA, fastest | `-Dchronos2.tornado.gemm=cublas-fp16` | 1.4–13.9× faster than the reference on an RTX 5070 Ti and an A10; forecasts move by up to 5e-3 of the series' range (see [VALIDATION.md](https://github.com/beestream-ai/jitCast-benchmarking/blob/main/docs/VALIDATION.md#chronos-2)) |
| **Apple Silicon** (tested: M4 Pro) | the defaults on Metal (`auto`: `simdgroup`; with an SDK that also has OpenCL, add `-Dtornado.metal.priority=100`) | Metal is faster than OpenCL on Apple GPUs |
| Other OpenCL devices | the defaults | Portable; not tuned |
