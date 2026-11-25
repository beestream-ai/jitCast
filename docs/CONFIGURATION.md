# Configuring jitcast

[← back to the README](../README.md)

Options are JVM properties (`-D…`), read when an engine is created. A value that does not parse, or a
GEMM mode the device cannot run, fails engine creation with the property's name; nothing falls back
silently. Shared options come first, then each model's own (`tirex.*`, `timestral.*`, `chronos2.*`, `timesfm.*`, `t0.*`, `toto.*`).

An engine can be shared between threads. A TornadoVM engine runs calls one at a time, because every
call of a shape reuses that shape's device buffers.

## Tuning by platform

The defaults are accurate to FP32 on every platform. Start from them and add flags from this table.

| Platform | TiRex-2 | Timestral | Chronos-2 | TimesFM 2.5 | t0 | Toto 2.0 |
|---|---|---|---|---|---|---|
| **NVIDIA, CUDA** (default: cuBLAS FP32 with CUDA graphs) | fastest: `-Dtirex.tornado.gemm=cublas-auto` | fastest: `-Dtimestral.tornado.gemm=cublas-fp16` | fastest: `-Dchronos2.tornado.gemm=cublas-fp16` | fastest: `-Dtimesfm.tornado.gemm=cublas-fp16` | fastest: `-Dt0.tornado.gemm=cublas-fp16` | fastest: `-Dtoto.tornado.gemm=cublas-fp16` |
| NVIDIA, closest to FP32 | `-Dtirex.tornado.slstmR=fp32` | defaults | defaults | defaults | defaults | defaults |
| NVIDIA, no vendor libraries | `-Dtirex.tornado.gemm=cutile` (toolkit 13.3+) or `half` | `-Dtimestral.tornado.gemm=half` | `-Dchronos2.tornado.gemm=half` | `-Dtimesfm.tornado.gemm=half` | `-Dt0.tornado.gemm=half` | `-Dtoto.tornado.gemm=half` |
| **Apple Silicon, Metal** | `-Dtirex.tornado.gemm=half` | defaults (simdgroup GEMMs) | defaults (simdgroup GEMMs) | defaults (simdgroup GEMMs) | defaults (simdgroup GEMMs) | defaults (simdgroup GEMMs) |
| Apple Silicon, SDK with Metal and OpenCL | as above, plus `-Dtornado.metal.priority=100` | `-Dtornado.metal.priority=100` | `-Dtornado.metal.priority=100` | `-Dtornado.metal.priority=100` | `-Dtornado.metal.priority=100` | `-Dtornado.metal.priority=100` |
| Other OpenCL GPUs | `-Dtirex.tornado.gemm=half` (`tiled` for FP32) | defaults | defaults | defaults | defaults | defaults |

The FP16 modes keep FP32 accumulation; their forecasts move slightly off the FP32 reference.

For every platform and model:

- **Device memory.** TornadoVM caps the memory it allocates at 4 GB unless
  `-Dtornado.device.memory` says otherwise (for example `20GB`). Raise it toward the GPU's size: the
  plan cache (below) uses it as its budget, and large checkpoints or batches need more than 4 GB.
- **Compiled plans.** Each new input shape compiles a plan. Raise `<model>.tornado.plans` when a
  workload cycles through more shapes than the cache keeps. For TiRex-2 with many different batch
  sizes, `-Dtirex.tornado.rowBuckets=pow2` lets sizes share plans; Timestral pads its shapes by
  default (`timestral.tornado.buckets`).
- **Batching.** One `forecast()` call with many series costs far less than many calls with one.
- **Long contexts.** Memory grows with rows × patches; `<model>.maxTokens` bounds one forward pass
  without changing the forecasts.

## Shared options

| Property | Default | Effect |
|---|---|---|
| `jitcast.tornado.device` | TornadoVM's default device | Device, `<backend>:<device>` as in `tornado --devices` |
| `tornado.device.memory` (TornadoVM) | `4GB` | Device memory TornadoVM may allocate; also the default plan budget |
| `tornado.metal.dispatch.spinWait` (TornadoVM) | `False`, set by jitcast | Metal spin-waits on each dispatch, which slows jitcast's many short kernels; jitcast turns it off. Code that calls TornadoVM before creating an engine should call `tornado.Devices.defaults()` first |
| `tornado.cuda.host.pinning` (TornadoVM) | `True`; `False` in the harness servers | Pinned host buffers. Turn off in applications that create and evict plans often: the churn can make CUDA allocations fail |
| `tornado.metal.priority` (TornadoVM) | — | `100` makes Metal backend `0` in an SDK that also has OpenCL; the simdgroup GEMMs need it |
| `tornado.cuda.nvcc` (TornadoVM) | — | `nvcc` with `tileiras`, for CUDA Tile (`cutile`) |

### Compiled plans

Each model compiles one plan per input shape and keeps the most recently used. The weights are on the
device once per engine: producer plans upload them, and every shape's plan reads them from there
(TornadoVM 8.2.0's buffers shared across execution plans), so a plan's footprint is its activations.
The cache keeps at most `<model>.tornado.plans` plans (8) within `<model>.tornado.planBytes` of device
memory (default: `tornado.device.memory` minus the weights), measured after each plan's first run, and
evicts the least recently used first. A plan larger than the budget still runs, alone. A plan whose first run
fails is dropped; if others were cached, they are evicted and the plan is retried once.

## Vendor libraries and CUDA features

| API | Used for | Enable with | Needs |
|---|---|---|---|
| cuBLAS SGEMM | FP32 / TF32 GEMMs | `<model>.tornado.gemm=cublas`, `cublas-tf32` | CUDA backend |
| cuBLAS GemmEx | FP16 tensor-core GEMMs | `cublas-fp16` (all), `cublas-auto` (TiRex-2) | CUDA backend |
| cuBLASLt | FP16 GEMMs | `-Dtirex.tornado.fp16Kernel=lt-h16` | CUDA backend |
| CUDA Tile | FP16 GEMM on tensor cores, written in TornadoVM | `-Dtirex.tornado.gemm=cutile` | CUDA Toolkit 13.3+ (`tileiras`), compute capability 8.0+, driver R580+ |
| CUDA graphs | replaying a plan | `<model>.tornado.cudaGraph` (on with cuBLAS) | CUDA backend |
| Apple simdgroup matrices | FP32 GEMMs | `<model>.tornado.gemm=simdgroup` (Timestral, Chronos-2) | Metal as TornadoVM's default backend |

## TiRex-2

| Property | Values (default first) | Effect |
|---|---|---|
| `tirex.tornado.gemm` | `auto`, `tiled`, `half`, `cublas`, `cublas-tf32`, `cublas-fp16`, `cublas-auto`, `cutile` | GEMM mode (below). `auto`: `cublas` on CUDA, `tiled` elsewhere |
| `tirex.tornado.fp16MinTokens` | `0` | `cublas-auto`: GEMMs with at least this many rows use FP16, smaller ones FP32. 0 is FP16 everywhere |
| `tirex.tornado.fp16Kernel` | `gemmex-mixed`, `gemmex-h16`, `lt-h16` | cuBLAS call for FP16 GEMMs: GemmEx FP16→FP32, GemmEx all-FP16, or cuBLASLt all-FP16 |
| `tirex.tornado.slstmR` | `auto`, `fp16-reg`, `fp16`, `fp32` | sLSTM recurrent weights: FP16 partly held in registers, FP16 streamed, or FP32 streamed. `auto` picks `fp16-reg` where it fits (see below). State is FP32 in all |
| `tirex.tornado.slstmRows` | `0` (auto) | Rows per sLSTM work-group; auto keeps 64 work-groups |
| `tirex.tornado.mlstm` | `auto`, `chunk`, `step` | mLSTM kernel: chunkwise, or one work-group per step. `auto` takes `chunk` unless its grid has fewer than 64 work-groups |
| `tirex.tornado.rowBuckets` | `off`, `pow2` | Pad row counts to powers of two so batch sizes share plans; the padding rows cost compute |
| `tirex.tornado.fuse` | `true`, `false` | Fused epilogues: bias + SiLU, residual + RMSNorm, sLSTM head norm + residual + RMSNorm |
| `tirex.tornado.cudaGraph` | `true` with `cublas*` / `cutile` | Replay each plan as a CUDA graph |
| `tirex.tornado.ffnHalf` | `true` | FP16 modes: keep FFN intermediates in FP16 only (about 15% less plan memory) |
| `tirex.tornado.plans` | `8` | Shapes kept compiled (rows, known covariates, rows in multi-row groups) |
| `tirex.tornado.planBytes` | `tornado.device.memory` minus the weights | Device memory the cached plans' activations may hold together |
| `tirex.maxTokens` | `131072` | Most tokens (rows × patches) per forward pass; bounds plan memory. `TiRexBench` ignores it unless set |
| `tirex.tornado.profile` | `false` | Print per-task kernel times after every run (analysis only) |
| `tirex.tornado.memReport` | `false` | Print each plan's buffers when it is built |

Programmatic: `ForecastPipeline(config, forward, batchSize, ttaDiff, ttaSignFlip)` sets series per
forward pass (512) and the TTA switches, `withBatchSize` and `withSignFlip` change them on an engine's
pipeline, and `TornadoForward(config, weights, maxCachedPlans)` sets the plan count.

### GEMM modes

| Mode | Kernel | Precision | Backends |
|---|---|---|---|
| `tiled` | `Kernels.gemm` | FP32 | all |
| `half` | `Kernels.gemmHalf` | FP16 in, FP32 accumulation | all |
| `cublas`, `cublas-tf32` | cuBLAS SGEMM | FP32, TF32 | CUDA |
| `cublas-fp16` | cuBLAS GemmEx | FP16 in, FP32 accumulation | CUDA |
| `cublas-auto` | GemmEx from `fp16MinTokens` rows, SGEMM below | mixed | CUDA |
| `cutile` | `Kernels.gemmHalfTile` (CUDA Tile) | FP16 in, FP32 accumulation | CUDA, toolkit 13.3+ |

### Kernel choices by device

`slstmR=fp16-reg` needs 512-thread work-groups with up to 128 registers per thread. It fits on CUDA
compute capability 12+ and on Apple GPUs, but not on an A100 (sm_80), where CUDA refuses the launch
and NVIDIA's OpenCL returns wrong results without an error. `auto` therefore uses it only on CUDA
12+ and Apple GPUs. OpenCL does not report the compute capability, so other OpenCL devices get the
streamed kernel.

Some kernels adapt to the device's largest work-group (256 on OpenCL on macOS, 1024 on Metal and
CUDA). The fused sLSTM norm kernel needs 512 threads and is skipped on smaller devices.

## Timestral

| Property | Values (default first) | Effect |
|---|---|---|
| `timestral.tornado.gemm` | `auto`, `simdgroup`, `blocked`, `tiled`, `half`, `cublas`, `cublas-tf32`, `cublas-fp16` | `auto`: `cublas` on CUDA, `simdgroup` when Metal is the default backend, `blocked` elsewhere. `blocked`: register-blocked FP32 from `blockedMinRows` rows, tiled below |
| `timestral.tornado.blockedMinRows` | `128` | Fewest GEMM rows for the register-blocked kernel |
| `timestral.tornado.wideMinRows` | `1024` | Fewest GEMM rows for its 64 × 128 tiles |
| `timestral.tornado.simdgroupWideMinRows` | `64` | Fewest GEMM rows for the 64 × 64 simdgroup kernel (32 × 32 below) |
| `timestral.tornado.splitKTarget` | `320` | Simdgroup GEMMs with fewer 64 × 64 tiles also split K across work-groups; `1` turns it off |
| `timestral.tornado.splitKMax` | `16` | Most K slices |
| `timestral.tornado.cudaGraph` | `true` with cuBLAS | Replay each plan as a CUDA graph |
| `timestral.tornado.attention` | `auto`, `kernel`, `cublas` | Time attention. `auto`, with a cuBLAS mode: cuBLAS batched GEMMs for blocks of at least `blasAttentionMinTokens` (128) tokens per row and at most `blasAttentionMaxRows` (8) rows, the kernel otherwise |
| `timestral.tornado.blasAttentionMinTokens` | `128` | `attention=auto`: the fewest tokens per row for the cuBLAS time attention |
| `timestral.tornado.blasAttentionMaxRows` | `8` | `attention=auto`: the most rows (series × variates) in a block for the cuBLAS time attention |
| `timestral.tornado.attentionScoreBytes` | `536870912` | Most memory the cuBLAS attention scores take at once; larger blocks run in chunks |
| `timestral.tornado.buckets` | `true` | Pad blocks to shared shapes so few plans compile; forecasts unchanged |
| `timestral.tornado.plans` | `8` | Block shapes kept compiled |
| `timestral.tornado.planBytes` | `tornado.device.memory` minus the weights | Device memory the cached plans' activations may hold together |
| `timestral.maxTokens` | `131072` | Most tokens (rows × patches) per forward pass |
| `timestral.tornado.profile`, `timestral.timing` | `false` | Print kernel times, or host and device phase times (analysis only; `timing` is read once per JVM) |

Programmatic: `TimestralPipeline.Options` (context cap, patches per block, cap to the observed range,
fed-back median threshold, sign-flip averaging, non-negative clipping), with the reference's defaults;
`withBatchSize` sets series per forward pass (512).

## Chronos-2

| Property | Values (default first) | Effect |
|---|---|---|
| `chronos2.tornado.gemm` | as `timestral.tornado.gemm` | Same modes and `auto` rule |
| `chronos2.tornado.blockedMinRows`, `wideMinRows`, `simdgroupWideMinRows`, `splitKTarget`, `splitKMax` | `128`, `1024`, `64`, `320`, `16` | As for Timestral |
| `chronos2.tornado.cudaGraph` | `true` with cuBLAS | Replay each plan as a CUDA graph |
| `chronos2.tornado.plans` | `8` | Shapes kept compiled (rows, context patches, output patches) |
| `chronos2.tornado.planBytes` | `tornado.device.memory` minus the weights | Device memory the cached plans' activations may hold together |
| `chronos2.maxTokens` | `131072` | Most tokens per forward pass, split between groups. A group is never split, so with cross-learning a batch can exceed it |
| `chronos2.tornado.profile` | `false` | Print per-task kernel times (analysis only) |

Programmatic: `Chronos2Pipeline.Options` (context cap, cross-learning, patches per pass, unrolled
quantile levels) with the reference's defaults; `withBatchSize` sets rows per batch (256).

## TimesFM 2.5

| Property | Values (default first) | Effect |
|---|---|---|
| `timesfm.tornado.gemm` | as `timestral.tornado.gemm` | Same modes and `auto` rule |
| `timesfm.tornado.blockedMinRows`, `wideMinRows`, `simdgroupWideMinRows`, `splitKTarget`, `splitKMax` | `128`, `1024`, `64`, `320`, `16` | As for Timestral |
| `timesfm.tornado.cudaGraph` | `true` with cuBLAS | Replay each plan as a CUDA graph |
| `timesfm.tornado.plans` | `8` | Shapes kept compiled (rows, prefill patches, decoding steps); each holds the key-value cache of its decoding steps |
| `timesfm.tornado.planBytes` | `tornado.device.memory` minus the weights | Device memory the cached plans' activations may hold together |
| `timesfm.maxTokens` | `131072` | Most tokens (series × patches, including the fed-back patches) per forward pass |
| `timesfm.tornado.profile` | `false` | Print per-task kernel times (analysis only) |

Programmatic: `TimesFmPipeline.Options` (the reference's `ForecastConfig`: context cap, input
normalization, continuous quantile head, flip averaging, non-negative clamping, quantile-crossing fix),
defaulting to the configuration Google recommends; `withBatchSize` sets series per forward pass (64).

## t0

| Property | Values (default first) | Effect |
|---|---|---|
| `t0.tornado.gemm` | as `timestral.tornado.gemm` | Same modes and `auto` rule |
| `t0.tornado.blockedMinRows`, `wideMinRows`, `simdgroupWideMinRows`, `splitKTarget`, `splitKMax` | `128`, `1024`, `64`, `320`, `16` | As for Timestral |
| `t0.tornado.cudaGraph` | `true` with cuBLAS | Replay each plan as a CUDA graph |
| `t0.tornado.plans` | `8` | Shapes kept compiled (rows, patches, output patches) |
| `t0.tornado.planBytes` | `tornado.device.memory` minus the weights | Device memory the cached plans' activations may hold together |
| `t0.maxTokens` | `16384` | Most tokens (rows × patches, including the forecast region and every rollout trajectory) per forward pass; a larger series runs alone |
| `t0.tornado.profile` | `false` | Print per-task kernel times (analysis only) |

There is no key-value cache: one forward pass covers up to 1,024 steps, and each step of the longer
rollout slides the window, which moves every position.

Programmatic: `T0Pipeline.Options` (the reference's `predict()` keywords: quantile levels, leading-missing
trimming, symmetric averaging, non-negative clipping, integer snapping, constant-covariate dropping),
with the reference's defaults; `withRecommendedAdjustments()` turns on the four adjustments it
recommends. `withBatchSize` sets series per forward pass (64), `withMaxTokens` the token budget.

## Toto 2.0

| Property | Values (default first) | Effect |
|---|---|---|
| `toto.tornado.gemm` | as `timestral.tornado.gemm` | Same modes and `auto` rule |
| `toto.tornado.blockedMinRows`, `wideMinRows`, `simdgroupWideMinRows` | `128`, `1024`, `64` | As for Timestral |
| `toto.tornado.cudaGraph` | `true` with cuBLAS | Replay each plan as a CUDA graph |
| `toto.tornado.plans` | `8` | Shapes kept compiled (rows, patches, forecast patches) |
| `toto.tornado.planBytes` | `tornado.device.memory` minus the weights | Device memory the cached plans' activations may hold together |
| `toto.tornado.profile` | `false` | Print per-task kernel times (analysis only) |

Programmatic: `TotoPipeline.Options` (context cap, the short-patch scaler fallback, the quantile cap), with
the defaults of Datadog's GluonTS predictor; `withBatchSize` sets series per forward pass (64).
