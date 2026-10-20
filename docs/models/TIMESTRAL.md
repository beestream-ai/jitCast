# Timestral

[← back to the README](../../README.md)

[Timestral](https://huggingface.co/mistralai/Timestral-5M) is Mistral AI's family of time-series
foundation models: 5M, 28M, 123M and 209M parameters, one architecture. jitcast reproduces
`TimestralForecaster.forecast()` from Mistral's [`timestral`](https://github.com/mistralai/timestral)
package end to end: 21-quantile forecasts for univariate and multivariate series, with past-only and
known-future covariates (numeric or categorical), any horizon (decoded block by block), and the clampsym test-time
augmentation.

| | |
|---|---|
| Architecture | patch transformer (32 steps per patch): time attention with partial RoPE and a sigmoid gate, attention across the variates of a series, SwiGLU MLPs, RMSNorm; a monotone 21-quantile head |
| Checkpoints | `mistralai/Timestral-5M`, `-28M`, `-123M`, `-209M` on Hugging Face, loaded as published (`config.json` + `model.safetensors`) |
| Validated | all four checkpoints on CPU, Metal, OpenCL and CUDA ([VALIDATION.md](../VALIDATION.md#timestral)) |
| Performance | [PERFORMANCE.md](../PERFORMANCE.md#timestral) |
| Engines | `timestral.tornado.TornadoTimestralEngine` (TornadoVM), `timestral.cpu.CpuTimestralEngine` (FP32 reference) |

## Quick start

```bash
# a checkpoint (any size; the weights' licence is below)
mkdir -p Timestral-5M && for f in config.json model.safetensors; do
  curl -L -o Timestral-5M/$f https://huggingface.co/mistralai/Timestral-5M/resolve/main/$f
done
```

```java
try (ForecastEngine engine = TornadoTimestralEngine.load(Path.of("Timestral-5M"))) {
    List<Forecast> forecasts = engine.forecast(List.of(
            TimeSeries.univariate(history),                              // NaN = missing
            new TimeSeries(targets, pastCovariates, futureCovariates),   // [V][T], [Vp][T], [Vf][T+H]
            new TimeSeries(targets, null, null, pastCodes, futureCodes)),// category codes: [Cp][T], [Cf][T+H]
            64);
    float[] median = forecasts.get(0).median(0);                         // values[target][quantile][step]
}
```

The forecaster's settings (context cap, block size, the cap to the observed range, sign-flip
averaging, non-negative clipping) are `TimestralPipeline.Options`, with the reference's defaults:

```java
TimestralPipeline p = engine.pipeline().withOptions(
        TimestralPipeline.Options.defaults(engine.config()).withSymmetricAveraging(false));
```

## Features

| Feature | Supported |
|---|---|
| Inputs | univariate series, several jointly forecast targets, past-only and known-future covariates, numeric and categorical, mixed batches, NaN gaps and late starts, series of different lengths in one batch |
| Context | up to the checkpoint's 32,768 steps (longer contexts keep the most recent) |
| Horizon | any: up to 32 patches (1,024 steps) per block, then block by block with the median fed back and each block in the advancing refine frame |
| Test-time augmentation | sign-flip averaging and clipping non-negative series at 0, both on by default as in the reference |
| Precision | FP32 throughout by default; optional FP16 or TF32 GEMM inputs |

## How it maps onto jitcast

| Reference | jitcast |
|---|---|
| `TimestralConfig` | `timestral.TimestralConfig` (unknown keys and unsupported settings fail the load) |
| `load_model` | `timestral.TimestralWeights`: every tensor bound by name and shape |
| `TimestralModel.predict_block` | `timestral.BlockForward`: `cpu.CpuTimestral`, `tornado.TimestralForward` |
| masks and positions (`masks.py`) | `timestral.BlockLayout`, shared by both engines |
| `scaler.py` | `timestral.Scaler` |
| `monotone_quantiles` | `timestral.Quantiles` |
| `TimestralForecaster.forecast` | `timestral.TimestralPipeline` |
| `covariate_encoding.target_encode` | `timestral.TargetEncoding` |

- **Host and device.** The scaler (float64 statistics, which Apple GPUs do not have), the patch
  tokens, the monotone quantile fan-out and the denormalization (which clamps infinities, which
  fast-math device compilers may drop) run on the host, in `timestral.BlockPrep`, with the rows in
  parallel. The patch embedding, the transformer and the head's projections run on the device as one
  task graph per block shape.
- **Rearranged weights.** The fused `q|k|v` projection is stored head-dim-major and becomes
  heads-major; the MLP's fused input projection, whose SiLU half is second, is swapped so the shared
  SwiGLU kernel applies.
- **Categorical covariates.** As in the reference, each category becomes the smoothed mean of the
  target where it occurs in the context, per series, on the host before the forward pass: `NaN` is a
  category of its own, a code the context never shows becomes the series mean, and a series with
  several targets uses its codes as they are. Codes are any finite numbers (`double`), so map strings
  to numbers first.
- **Covariates.** With covariates the reference also runs a decoder token for every context patch.
  Those tokens attend only to themselves and nothing reads them, so jitcast leaves them out; the
  forecasts are the same.
- **Kernels.** Token assembly, RoPE, time and variate attention, the gate and the decoder gather are
  Timestral's own ([KERNELS.md](../KERNELS.md#timestral)); the GEMMs, RMSNorm and SwiGLU are shared.

## Options

| Property | Default | Effect |
|---|---|---|
| `jitcast.tornado.device` | TornadoVM's default | The device, `<backend>:<device>` |
| `timestral.tornado.gemm` | `auto` | `auto`: `cublas` on CUDA, `simdgroup` on Metal when Metal is TornadoVM's default backend, `blocked` elsewhere. `simdgroup`: FP32 GEMM on Apple's simdgroup matrix units (Metal, TornadoVM 7.3.1+); `blocked`: TornadoVM's register-blocked FP32 GEMM from `blockedMinRows` rows and the tiled one under it; `tiled`; `half`: TornadoVM's FP16 GEMM (FP32 accumulation); `cublas`, `cublas-tf32`, `cublas-fp16` (CUDA) |
| `timestral.tornado.blockedMinRows` | `128` | The fewest GEMM rows that use the register-blocked kernel |
| `timestral.tornado.simdgroupWideMinRows` | `64` | The fewest GEMM rows that use the 64 × 64 simdgroup kernel (the 32 × 32 one below) |
| `timestral.tornado.splitKTarget` | `320` | A 64 × 64 simdgroup GEMM with fewer tiles than this also splits K across work-groups (`1`: off) |
| `timestral.tornado.splitKMax` | `16` | The most K slices of a split GEMM |
| `timestral.tornado.cudaGraph` | `true` with cuBLAS | Capture the task graph once and replay it as a CUDA graph |
| `timestral.tornado.attention` | `auto` | Time attention: `auto` uses cuBLAS FP32 batched GEMMs with a cuBLAS GEMM mode for blocks of at least `blasAttentionMinTokens` (128) tokens per row and at most `blasAttentionMaxRows` (8) rows, TornadoVM's attention kernel otherwise; `cublas` or `kernel` force one |
| `timestral.tornado.attentionScoreBytes` | `536870912` (512 MB) | The most memory the cuBLAS time attention's scores take at once |
| `timestral.tornado.plans` | `8` | Block shapes kept compiled (least recently used are evicted) |
| `timestral.tornado.planBytes` | `8589934592` (8 GiB) | The most host memory the cached plans' buffers take together; the least recently used plans are evicted first |
| `timestral.tornado.profile` | `false` | Print TornadoVM's per-task kernel times after every execution |
| `timestral.timing` | `false` | Print the time of each host and device phase per forecast (analysis only) |
| `timestral.maxTokens` | `131072` | The most tokens one forward pass takes; forecasts do not depend on the split |

With cuBLAS the weights are held twice on the device (the packed buffer and one array per GEMM
matrix), about 0.8 GB more for the 209M checkpoint; the FP16 modes add an FP16 copy of the GEMM
weights.

## Limits

- Every checkpoint has 64-wide attention heads; the attention kernels support up to 128, in
  multiples of 8.

## Licence

The Timestral weights are Mistral AI's, under a Modified MIT licence that withholds every right from
companies whose monthly revenue exceeds $20 million; read it before use. Mistral's inference code is
Apache 2.0. jitcast does not distribute the weights.
