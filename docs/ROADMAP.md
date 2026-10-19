# Roadmap

[← back to the README](../README.md)

Planned work on jitcast, and the TornadoVM issues that limit it today. The current results are in
[PERFORMANCE.md](PERFORMANCE.md) and [VALIDATION.md](VALIDATION.md).

## Models

1. **Chronos-2** (Amazon): in progress. Its time and cross-variate attention are close to Timestral's.
2. **Further candidates**: TimesFM, Toto, Moirai 2, TTM, Sundial, checked against the GIFT-Eval
   leaderboard and each model's weight licence before porting.
3. **Benchmark parity for Timestral**: GIFT-Eval and fev-bench scored through Mistral's or the
   benchmarks' own harnesses, as for TiRex-2.

## Engine

1. **One GEMM layer for every family.** Timestral selects its GEMMs through the shared `GemmMode`
   (TornadoVM FP32 and FP16 kernels, cuBLAS); TiRex-2 still has its own dispatch, with its FP16 twins,
   cuBLASLt and CUDA Tile paths. Moving TiRex-2 onto the shared layer also gives it the register-blocked
   FP32 GEMM, which is 2× faster than `tiled` on Apple GPUs.
2. **Timestral's host steps on the device where it has float64** (CUDA, OpenCL with fp64): the scaler,
   the quantile head's fan-out and denormalization, the refine frames and the cap. With the rows in
   parallel they still cost about 100 ms per forecast at 512 series on the A100's host, against about
   66 ms of device time for Timestral-5M and 265 ms for 28M.
3. **One copy of the weights with cuBLAS.** cuBLAS library tasks take whole arrays, so every GEMM
   weight is held twice (packed, and one array per matrix): about 0.8 GB more for Timestral-209M and
   300 MB for TiRex-2.
4. **One plan for every batch size.** Each new batch shape builds its own plan (kernel compilation,
   buffer allocation, weight upload, CUDA graph capture: 0.4–2 s), and `TiRexServer` restarts after
   `--max-shapes` shapes. Allocating the buffers once for the largest batch and passing the row count
   at run time removes the rebuilds; padding Timestral's contexts with leading empty patches to shared
   sizes would do the same for its block shapes.
5. **Share the weights across plans.** Every cached plan uploads its own weights (about 630 MB for
   TiRex-2), which limits how many large plans fit on a GPU. This needs device buffers shared across
   TornadoVM execution plans (see below).
6. **Sign-flip TTA in one batch.** Both models forecast the series and its negation as two forward
   passes; concatenating them into one halves the kernel launches.
7. **Faster portable FP16 GEMM** (`half`): vectorised half2 loads, a larger K slice, double buffering.
8. **Smaller items.**
   - Fuse the SwiGLU activation into a GEMM (about 11% of GPU time at 256 series with TiRex-2's
     `cublas-auto`).
   - Lower host memory: drop the host weight copies after upload, or map them straight from the
     safetensors file.
   - TiRex-2: re-check the server's restart guard (`--max-shapes`) on TornadoVM 7.1.0.
   - Timestral: categorical covariates (target encoding on the host, as in the reference).

## Evaluation

- TiRex-2: re-measure Metal and OpenCL on TornadoVM 7.1.0; GIFT-Eval parity in multivariate mode.
- CUDA Tile (`cutile`), once TornadoVM can compile it, and then tuning its GEMM: it was about 5×
  slower than cuBLAS per GEMM when it last ran.
- `torch.compile` baselines for Timestral (the current reference runs are eager).

## TornadoVM issues that affect jitcast

| Issue | Effect on jitcast | Workaround in jitcast |
|---|---|---|
| The Tile API's `TileContext.checkShape` cannot be compiled into a kernel (its message uses string concatenation) | `gemm=cutile` fails on TornadoVM 7.1.0 | use `cublas-auto` or `half` |
| Device numbering differs between JVMs launched with and without the test profile, on SDKs with several backends | the accelerated tests can pick a different backend than the benchmarks | set `-Djitcast.tornado.device` per JVM (`TEST_CUDA_DEVICE` in the evaluation scripts) |
| A kernel that cannot launch for lack of registers fails silently on NVIDIA's OpenCL | wrong results instead of an error | `slstmR=auto` avoids the register-resident sLSTM kernel there |
| Metal's dispatch spin wait | up to 2× slower on Metal | jitcast turns it off by default |
| Simdgroup matrices that meet after an if/else are merged wrongly (Metal): the copy from one branch runs on every path | a kernel that keeps its accumulators across such a join gets the other branch's values | `gemmSimdgroupWide` stores each branch's accumulators to threadgroup memory before the join |
| No float64 on Apple GPUs | statistics that need float64 run on the host | Timestral's scaler and refine step run on the host |
| Fast math folds NaN checks away (Metal) | in-kernel NaN tests do not work | the host uploads NaN as 0 plus a mask |
| The cuBLAS provider prefers CUDA 12 libraries over CUDA 13 | slow fallback GEMM kernels on newer GPUs | a library shim directory ([INSTALL.md](INSTALL.md#nvidia-extras)) |
| cuBLAS library tasks take whole arrays, with no element offsets | each GEMM weight needs its own array (about 300 MB extra) | separate weight arrays |
| No device buffers shared across execution plans | weights are uploaded once per cached plan | fewer cached plans (`--plans`) |
| Grid sizes are not checked against device limits, and errors lose their cause | launch failures are hard to diagnose | — |
| Cleanup after a failed device allocation frees buffers that were never allocated | the server can fail on shutdown after an out-of-memory error | `TiRexServer` catches it on close |
