# Chronos-2

[← back to the README](../../README.md)

[Chronos-2](https://huggingface.co/amazon/chronos-2) is Amazon's time-series foundation model.
jitcast reproduces `Chronos2Pipeline.predict()` from Amazon's
[`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting) package end to end:
quantile forecasts for univariate and multivariate series, with past-only and known-future
covariates, cross-learning across a batch, and any horizon.

| | |
|---|---|
| Architecture | T5-style encoder over 16-step patches: time attention with RoPE along each series, group attention across a group's series at each patch, ReLU feed-forward layers, RMSNorm; a `[REG]` token and one token per forecast patch; residual-MLP patch embedding and quantile head |
| Checkpoints | `amazon/chronos-2` (120M, 21 quantiles), `autogluon/chronos-2-small` (28M, 13 quantiles), loaded as published |
| Validated | both checkpoints on CPU, Metal, OpenCL and CUDA |
| Engines | `chronos2.tornado.TornadoChronos2Engine` (TornadoVM), `chronos2.cpu.CpuChronos2Engine` (FP32 reference) |

## Quick start

```bash
mkdir -p chronos-2 && for f in config.json model.safetensors; do
  curl -L -o chronos-2/$f https://huggingface.co/amazon/chronos-2/resolve/main/$f
done
```

```java
try (ForecastEngine engine = TornadoChronos2Engine.load(Path.of("chronos-2"))) {
    List<Forecast> forecasts = engine.forecast(List.of(
            TimeSeries.univariate(history),                              // NaN = missing
            new TimeSeries(targets, pastCovariates, futureCovariates)),  // [V][T], [Vp][T], [Vf][T+H]
            64);
    float[] median = forecasts.get(0).median(0);                         // values[target][quantile][step]
}
```

Pipeline settings are `Chronos2Pipeline.Options`, with the reference's defaults:

```java
Chronos2Pipeline p = engine.pipeline().withOptions(
        Chronos2Pipeline.Options.defaults(engine.config()).withCrossLearning(true));
```

## Features

| Feature | Supported |
|---|---|
| Inputs | univariate, several joint targets, past-only and known-future numeric covariates, mixed batches, NaN gaps and late starts, ragged lengths |
| Cross-learning | one group per batch, as the reference's `cross_learning=True`; 256 rows per batch by default |
| Context | up to 8,192 steps (longer contexts keep the most recent) |
| Horizon | any: up to 1,024 steps per forward pass, then the reference's long-horizon unrolling |
| Precision | FP32 by default; optional FP16 or TF32 GEMM inputs |

## How it maps onto jitcast

| Reference | jitcast |
|---|---|
| `Chronos2CoreConfig`, `Chronos2ForecastingConfig` | `chronos2.Chronos2Config` (unknown keys and unsupported settings fail the load) |
| `Chronos2Model` parameters | `chronos2.Chronos2Weights`: every tensor bound by name and shape |
| `Chronos2Model.forward` | `chronos2.ModelForward`: `cpu.CpuChronos2`, `tornado.Chronos2Forward` |
| masks and positions (`Chronos2Encoder`) | `chronos2.ModelLayout`, shared by both engines |
| `InstanceNorm`, patching, time encoding, inverse | `chronos2.ModelPrep` |
| `interpolate_quantiles`, `weighted_quantile` | `chronos2.QuantileMath` |
| `Chronos2Pipeline.predict`, `Chronos2Dataset` | `chronos2.Chronos2Pipeline` |

- **Forward pass.** Each row (a target or covariate) has one token per context patch, the `[REG]`
  token and one token per forecast patch. Each block runs time attention along a row (bidirectional,
  over observed tokens), group attention across a group's rows at each token, and a ReLU
  feed-forward layer. As in the reference, attention scores are not scaled by `1/√d_kv`.
- **Host and device.** Instance norm, patch tokens and the inverse norm run on the host in
  `chronos2.ModelPrep`, rows in parallel. The patch embedding, encoder and output embedding run on
  the device, one plan per shape. The device engine makes each group a contiguous run of rows for the
  group-attention kernel and restores the order after.
- **Long horizons.** Past 1,024 steps the pipeline unrolls as the reference does: each row is repeated
  per unrolled level (0.1 … 0.9), each copy's context extended with that level of the forecast, and
  the next pass's forecasts merged back into quantiles as a weighted sample.
- **Kernels.** Token assembly, RoPE, time and group attention and the future-token gather are
  Chronos-2's own ([KERNELS.md](../KERNELS.md#chronos-2)); GEMMs (ReLU epilogue) and RMSNorm are shared.

## Options and tuning

`chronos2.*` options are in [CONFIGURATION.md](../CONFIGURATION.md#chronos-2). The defaults use
cuBLAS FP32 with CUDA graphs on NVIDIA and simdgroup GEMMs on Metal; `-Dchronos2.tornado.gemm=cublas-fp16`
is fastest on NVIDIA.

## Limits

- Categorical covariates are not supported: a series with them fails the call. Encode them as
  numbers first.
- Only float32 checkpoints with ReLU feed-forward layers, non-overlapping patches and equal input and
  output patch sizes load; every published checkpoint qualifies. On the device, heads must be 16 to
  64 wide, in multiples of 16.
- Validated on `chronos-2` and `chronos-2-small`. `autogluon/chronos-2-synth` shares the architecture
  but has no goldens.
- NaN in a forecast is not reported as an error.

## Licence

The Chronos-2 weights and Amazon's inference code are under the Apache License 2.0. jitcast does not
distribute the weights.
