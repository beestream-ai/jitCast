# Roadmap

[← back to the README](../README.md)

Planned work on jitcast, and the TornadoVM limits that affect it today.

## Models

- **Chronos-2:** categorical covariates (target-encoded in the reference, as for Timestral).
- **New families:** TimesFM, Toto, Moirai 2, TTM and Sundial are candidates, each checked against the
  GIFT-Eval leaderboard and its weight licence before porting.
- **Timestral's long-context attention:** its time attention kernel walks keys one at a time. The
  blocked, register-tiled structure of Chronos-2's kernel would apply.

## Engine

- **One GEMM layer for every family.** Timestral and Chronos-2 share `tornado.GemmMode`; TiRex-2
  still has its own dispatch (FP16 twins, cuBLASLt, CUDA Tile). Moving it over also gives TiRex-2
  the register-blocked and simdgroup FP32 GEMMs.
- **One plan for every batch size.** Each new batch shape builds a plan (compilation, allocation,
  weight upload, CUDA graph capture). Allocating once for the largest batch and passing the row count
  at run time would remove the rebuilds.
- **Weights shared across plans.** Every cached plan holds its own copy of the weights, which limits
  how many large plans fit on a GPU. This needs device buffers shared across TornadoVM execution
  plans.
- **One copy of the weights with cuBLAS.** cuBLAS tasks take whole arrays, so each GEMM weight is
  held twice: packed, and as its own array.
- **Timestral's host steps on the device** where it has float64 (CUDA, OpenCL with fp64): the
  scaler, the quantile fan-out and denormalization, the refine frames and the cap.
- **Sign-flip TTA in one batch.** TiRex-2 and Timestral run a series and its negation as two passes;
  one concatenated pass halves the kernel launches.
- **Smaller items:** a faster portable FP16 GEMM (`half`: vector loads, larger K slices, double
  buffering); SwiGLU fused into its GEMM; lower host memory by dropping host weight copies after
  upload; tuning the CUDA Tile GEMM.

## TornadoVM limits that affect jitcast

| Limit | Effect on jitcast | Workaround |
|---|---|---|
| No device buffers shared across execution plans | each cached plan uploads its own weights | the plan cache counts weights against its budget |
| cuBLAS tasks take whole arrays, without offsets | each GEMM weight needs its own array | separate weight arrays |
| TC (TornadoVM 8) does not unroll loops, so per-thread arrays indexed in loops stay in memory | slow attention and sLSTM kernels | kernels keep accumulators in named scalars |
| Metal spin-waits on each dispatch | slow Metal runs | jitcast turns it off |
| Fast math folds NaN checks away (Metal) | in-kernel NaN tests fail | the host uploads NaN as 0 plus a mask |
| No float64 on Apple GPUs | float64 statistics run on the host | Timestral's scaler and refine step run on the host |
| Launch failures from register pressure go unchecked on NVIDIA's OpenCL | wrong results instead of an error | `slstmR=auto` avoids the register-resident sLSTM kernel there |
| Device numbering can differ between JVMs with and without the test profile, on multi-backend SDKs | tests and applications can pick different backends | set `-Djitcast.tornado.device` per JVM |
| Grid sizes are not checked against device limits; errors lose their cause | launch failures are hard to diagnose | — |
| Cleanup after a failed allocation frees buffers that were never allocated | shutdown can fail after an out-of-memory error | the servers catch it on close |

Fixed in TornadoVM 8.0.0, so jitcast needs no workaround there: the CUDA Tile `checkShape`
compile failure, the runtime assertion on a backend's second device, simdgroup matrices merged wrongly
after an if/else on Metal, and cuBLAS preferring CUDA 12 libraries.
