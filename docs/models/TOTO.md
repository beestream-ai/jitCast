# Toto 2.0

[← back to the README](../../README.md)

[Toto 2.0](https://huggingface.co/collections/Datadog/toto-20) is Datadog's family of time-series
foundation models, from 4M to 2.5B parameters. jitcast reproduces Datadog's
[`toto2`](https://github.com/DataDog/toto/tree/main/toto2) package end to end as its GluonTS predictor
runs it: 9-quantile forecasts for univariate and multivariate series with past covariates, in one
forward pass.

| | |
|---|---|
| Architecture | decoder-only transformer over 32-step patches, trained with unit scaling (u-μP): a residual-MLP embedding of each patch's values and missing flags, pre-norm layers that attend across time (causal, with a rotary embedding and xPos scaling on half of each head) except the last, which attends across the variates of a series, SwiGLU feed-forward layers, and a residual-MLP head of 9 quantiles for the next patch |
| Checkpoints | `Datadog/Toto-2.0-4m`, `-22m`, `-313m`, `-1B`, `-2.5B`, loaded as published (`model.safetensors` + `config.json`) |
| Validated | CPU (4m, 22m, 313m) and CUDA |
| Engines | `toto.tornado.TornadoTotoEngine` (TornadoVM), `toto.cpu.CpuTotoEngine` (FP32 reference) |

## Quick start

```bash
mkdir -p toto-2.0-22m && for f in config.json model.safetensors; do
  curl -L -o toto-2.0-22m/$f https://huggingface.co/Datadog/Toto-2.0-22m/resolve/main/$f
done
```

```java
try (ForecastEngine engine = TornadoTotoEngine.load(Path.of("toto-2.0-22m"))) {
    List<Forecast> forecasts = engine.forecast(List.of(TimeSeries.univariate(history)), 96);  // NaN = missing
    float[] median = forecasts.get(0).median(0);                                             // values[target][quantile][step]
}
```

The forecasting options are `TotoPipeline.Options`. The defaults are those of Datadog's
`Toto2GluonTSModelConfig` and its leaderboard evaluations: a 4,096-step context, the short-patch scaler
fallback and the quantile cap.

```java
TotoPipeline p = engine.pipeline().withOptions(TotoPipeline.Options.defaults().withMaxContext(2048));
```

## Features

| Feature | Supported |
|---|---|
| Inputs | series of any length with NaN gaps and late starts (forward-filled and masked, as the reference), ragged batches |
| Multivariate | the targets of a series and its past covariates attend to each other in the variate layer; forecasts for the targets |
| Context | up to `maxContext` steps (default 4,096) minus the horizon's patches but one; longer contexts keep the most recent |
| Horizon | any; every forecast patch is predicted in the same forward pass |
| Options | `maxContext`, `minObservations` (the scaler fallback, default 8), `quantileCap` (default 10,000) |
| Precision | FP32 by default; optional FP16 or TF32 GEMM inputs |

## How it maps onto jitcast

| Reference | jitcast |
|---|---|
| `Toto2ModelConfig`, `config.json` | `toto.TotoConfig` (unknown keys and unsupported settings fail the load) |
| `Toto2Model` parameters | `toto.TotoWeights`: every tensor bound by name and shape |
| `Toto2Model.forward` after the scaler | `toto.TotoForward`: `cpu.CpuToto`, `tornado.TornadoTotoForward` |
| `PatchedCausalStdScaler`, `backfill_short_patches`, `Toto2Model.forecast` | `toto.TotoPipeline` |
| `Toto2GluonTSModel` (imputation, instance splitter) | `toto.TotoPipeline` |

- **Host and device.** Imputation, trimming, the causal scaler (in double precision, as the
  reference) and the real-space steps (`sinh`, the cap, sorting) run on the host. The patch embedding,
  the transformer and the head run on the device.
- **Unit scaling.** The u-μP constants (each linear layer's fan-in scale, the residual weights, the
  query's per-dimension scale) are folded into the packed weights, so the device runs plain linear
  layers.
- **Padding.** The reference pads every context to its full window; jitcast pads to whole patches only.
  The forecasts are the same: positions enter attention only relatively, and padding patches see only
  each other.
- **Kernels.** The rotary embedding, the time and variate attention, the forecast-patch gather and the
  residual mixing are Toto's own ([KERNELS.md](../KERNELS.md#toto-20)); the GEMMs, RMSNorm and SwiGLU are
  shared.

## Options and tuning

`toto.*` options are in [CONFIGURATION.md](../CONFIGURATION.md#toto-20). The defaults use cuBLAS FP32 with
CUDA graphs on NVIDIA and simdgroup GEMMs on Metal; `-Dtoto.tornado.gemm=cublas-fp16` uses FP16 tensor
cores on NVIDIA.

## Limits

- Known-future and categorical covariates are not supported (Datadog plans exogenous variables for a
  later release); a series with them fails the call.
- Block decoding with a key-value cache (`decode_block_size`) is not implemented; forecasts use the single
  forward pass of Datadog's evaluations.
- The 1B and 2.5B checkpoints load with the same code but have not been validated against the reference
  yet.

## Licence

The Toto 2.0 weights and Datadog's inference code are under the Apache License 2.0. jitcast does not
distribute the weights.
