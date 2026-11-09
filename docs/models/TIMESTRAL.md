# Timestral

[← back to the README](../../README.md)

[Timestral](https://huggingface.co/mistralai/Timestral-5M) is Mistral AI's family of time-series
foundation models: 5M, 28M, 123M and 209M parameters, one architecture. jitcast reproduces
`TimestralForecaster.forecast()` from Mistral's [`timestral`](https://github.com/mistralai/timestral)
package end to end: 21-quantile forecasts for univariate and multivariate series, with past-only and
known-future covariates (numeric or categorical), any horizon, and the reference's test-time
augmentation.

| | |
|---|---|
| Architecture | patch transformer (32 steps per patch): time attention with partial RoPE and a sigmoid gate, attention across a series' variates, SwiGLU MLPs, RMSNorm; a monotone 21-quantile head |
| Checkpoints | `mistralai/Timestral-5M`, `-28M`, `-123M`, `-209M`, loaded as published (`config.json` + `model.safetensors`) |
| Validated | all four checkpoints on CPU, Metal, OpenCL and CUDA |
| Engines | `timestral.tornado.TornadoTimestralEngine` (TornadoVM), `timestral.cpu.CpuTimestralEngine` (FP32 reference) |

## Quick start

```bash
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

Forecaster settings are `TimestralPipeline.Options`, with the reference's defaults:

```java
TimestralPipeline p = engine.pipeline().withOptions(
        TimestralPipeline.Options.defaults(engine.config()).withSymmetricAveraging(false));
```

## Features

| Feature | Supported |
|---|---|
| Inputs | univariate, several joint targets, past-only and known-future covariates (numeric and categorical), mixed batches, NaN gaps and late starts, ragged lengths |
| Context | up to 32,768 steps (longer contexts keep the most recent) |
| Horizon | any: up to 1,024 steps per block, then block by block with the median fed back |
| Test-time augmentation | sign-flip averaging and non-negative clipping, on by default as in the reference |
| Precision | FP32 by default; optional FP16 or TF32 GEMM inputs |

## How it maps onto jitcast

| Reference | jitcast |
|---|---|
| `TimestralConfig` | `timestral.TimestralConfig` (unknown keys and unsupported settings fail the load) |
| `load_model` | `timestral.TimestralWeights`: every tensor bound by name and shape |
| `TimestralModel.predict_block` | `timestral.BlockForward`: `cpu.CpuTimestral`, `tornado.TimestralForward` |
| `masks.py` | `timestral.BlockLayout`, shared by both engines |
| `scaler.py` | `timestral.Scaler` |
| `monotone_quantiles` | `timestral.Quantiles` |
| `TimestralForecaster.forecast` | `timestral.TimestralPipeline` |
| `covariate_encoding.target_encode` | `timestral.TargetEncoding` |

- **Host and device.** The scaler (float64, which Apple GPUs lack), patch tokens, quantile fan-out
  and denormalization run on the host in `timestral.BlockPrep`, rows in parallel. The patch
  embedding, transformer and head projections run on the device, one plan per block shape.
- **Rearranged weights.** The fused q|k|v projection becomes heads-major; the MLP input projection's
  halves are swapped so the shared SwiGLU kernel applies.
- **Categorical covariates.** As in the reference, each category becomes the smoothed target mean
  where it occurs in the context, per series, on the host. `NaN` is its own category; an unseen code
  becomes the series mean. Codes are any finite numbers, so map strings to numbers first.
- **Covariates.** The reference also runs a decoder token per context patch when covariates are
  present. Nothing reads those tokens, so jitcast skips them; forecasts are the same.
- **Kernels.** Token assembly, RoPE, time and variate attention, the gate and the decoder gather are
  Timestral's own ([KERNELS.md](../KERNELS.md#timestral)); GEMMs, RMSNorm and SwiGLU are shared.

## Options and tuning

`timestral.*` options are in [CONFIGURATION.md](../CONFIGURATION.md#timestral). The defaults use
cuBLAS FP32 with CUDA graphs on NVIDIA and simdgroup GEMMs on Metal; `-Dtimestral.tornado.gemm=cublas-fp16`
is fastest on NVIDIA. With cuBLAS each plan holds the GEMM weights twice (packed and per matrix), and
the FP16 modes add an FP16 copy; raise `tornado.device.memory` for the larger checkpoints.

## Limits

- Attention heads up to 128 wide, in multiples of 8 (every checkpoint has 64).

## Licence

The Timestral weights are Mistral AI's, under a Modified MIT licence that withholds every right from
companies whose monthly revenue exceeds $20 million; read it before use. Mistral's inference code is
Apache 2.0. jitcast does not distribute the weights.
