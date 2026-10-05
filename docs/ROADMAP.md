# Roadmap

[← back to the README](../README.md)

Planned work on jitrex, and the TornadoVM issues that limit it today. The current results are in [PERFORMANCE.md](PERFORMANCE.md) and [VALIDATION.md](VALIDATION.md).

## Engine: performance and memory

1. **One plan for every batch size.** Today each new batch shape builds its own plan (kernel compilation, buffer allocation, weight upload, CUDA graph capture: 0.4–2 s), and `TiRexServer` restarts after `--max-shapes` shapes. Allocating the buffers once for the largest batch and passing the row count at run time removes the rebuilds, the memory churn and the restarts.
2. **Share the weights across plans.** Every cached plan uploads its own weights and FP16 copies (about 630 MB), which limits how many large long-context plans fit on a GPU. This needs device buffers shared across TornadoVM execution plans (see below).
3. **Re-check the server's restart guard on TornadoVM 7.1.0.** `--max-shapes 12` guards against cuBLAS handles leaking across plans; the plan-freeing fixes in TornadoVM 7.1.0 may make it unnecessary, which would remove the restarts from fev-bench and GIFT-Eval runs.
4. **Sign-flip TTA in one batch.** The GIFT-Eval checkpoint forecasts each batch twice, on the series and on its negation; concatenating both into one forward pass halves the kernel launches.
5. **Faster portable FP16 GEMM** (`half`): vectorised half2 loads, a larger K slice, double buffering. It is the fast path on Metal and OpenCL, and on an A100 it falls behind NXAI at large batches (0.5–0.7×), where cuBLAS is 3–6× faster.
6. **Smaller items.**
   - Fuse the SwiGLU activation into a GEMM (about 11% of GPU time at 256 series with `cublas-auto`).
   - A register-blocked FP32 portable GEMM (`tiled`).
   - Lower host memory (3–9.6 GB resident, against 2 GB for NXAI): drop the host weight copies after upload, or map them straight from the safetensors file.
   - Split batches by a rows × patches budget rather than by series count, so that covariate-heavy series do not run out of memory.

## Evaluation

- Re-measure Metal and OpenCL (Apple M4 Pro, and OpenCL on NVIDIA) on TornadoVM 7.1.0; they are validated but have no current results.
- GIFT-Eval parity in multivariate mode (the current GIFT-Eval run is univariate).
- CUDA Tile (`gemm=cutile`), once TornadoVM can compile it, and then tuning its GEMM: it was about 5× slower than cuBLAS per GEMM when it last ran.

## TornadoVM issues that affect jitrex

| Issue | Effect on jitrex | Workaround in jitrex |
|---|---|---|
| The Tile API's `TileContext.checkShape` cannot be compiled into a kernel (its message uses string concatenation) | `gemm=cutile` fails on TornadoVM 7.1.0 | use `cublas-auto` or `half` |
| Device numbering differs between JVMs launched with and without the test profile, on SDKs with several backends | the accelerated tests can pick a different backend than the benchmarks | set `-Dtirex.tornado.device` per JVM (`TEST_CUDA_DEVICE` in the evaluation scripts) |
| A kernel that cannot launch for lack of registers fails silently on NVIDIA's OpenCL | wrong results instead of an error | `slstmR=auto` avoids the register-resident sLSTM kernel there |
| Metal's dispatch spin wait | up to 2× slower on Metal | jitrex turns it off by default |
| Fast math folds NaN checks away (Metal) | in-kernel NaN tests do not work | the host uploads NaN as 0 plus a mask |
| The cuBLAS provider prefers CUDA 12 libraries over CUDA 13 | slow fallback GEMM kernels on newer GPUs | a library shim directory ([INSTALL.md](INSTALL.md#nvidia-extras)) |
| cuBLAS library tasks take whole arrays, with no element offsets | each GEMM weight needs its own array (about 300 MB extra) | separate weight arrays |
| No device buffers shared across execution plans | weights are uploaded once per cached plan | fewer cached plans (`--plans`) |
| Grid sizes are not checked against device limits, and errors lose their cause | launch failures are hard to diagnose | — |
| Cleanup after a failed device allocation frees buffers that were never allocated | the server can fail on shutdown after an out-of-memory error | `TiRexServer` catches it on close |
