# TimesFM 2.5

[← back to the README](../../README.md)

[TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) is Google Research's
decoder-only time-series foundation model. jitcast reproduces `TimesFM_2p5_200M_torch.forecast()` from
Google's [`timesfm`](https://github.com/google-research/timesfm) package end to end: point and
9-quantile forecasts for univariate series, with the reference's forecasting options.

| | |
|---|---|
| Architecture | decoder-only transformer over 32-step patches: a residual-MLP tokenizer of each patch's values and mask, 20 causal layers (RoPE, per-head RMSNorm of q and k, a learned per-dimension query scale, SiLU feed-forward layers, RMSNorm before and after each sublayer), a point head of 128 steps per patch and a continuous quantile head of 1,024 steps |
| Checkpoints | `google/timesfm-2.5-200m-pytorch` (200M, 9 quantiles), loaded as published (`model.safetensors` + `config.json`) |
| Validated | CPU, CUDA, and Metal and OpenCL on Apple Silicon |
| Engines | `timesfm.tornado.TornadoTimesFmEngine` (TornadoVM), `timesfm.cpu.CpuTimesFmEngine` (FP32 reference) |

## Quick start

```bash
mkdir -p timesfm-2.5-200m && for f in config.json model.safetensors; do
  curl -L -o timesfm-2.5-200m/$f https://huggingface.co/google/timesfm-2.5-200m-pytorch/resolve/main/$f
done
```

```java
try (ForecastEngine engine = TornadoTimesFmEngine.load(Path.of("timesfm-2.5-200m"))) {
    List<Forecast> forecasts = engine.forecast(List.of(TimeSeries.univariate(history)), 64);  // NaN = missing
    float[] median = forecasts.get(0).median(0);                                             // values[target][quantile][step]
}
```

The forecasting options are `TimesFmPipeline.Options`, the reference's `ForecastConfig`. The defaults are
the configuration Google's documentation recommends: a 1,024-step context, input normalization, the
continuous quantile head, flip averaging, non-negative clamping and quantile-crossing fixes.

```java
TimesFmPipeline p = engine.pipeline().withOptions(
        TimesFmPipeline.Options.defaults(engine.config()).withMaxContext(4096));
```

## Features

| Feature | Supported |
|---|---|
| Inputs | univariate series of any length, NaN gaps and late starts (interpolated, as the reference), ragged batches; each target of a multi-target series is forecast on its own |
| Context | up to `maxContext` steps (default 1,024; the model takes up to 16,384 with the horizon); longer contexts keep the most recent |
| Horizon | any, 128 steps per decoding step; up to 1,024 with the continuous quantile head |
| Options | `maxContext`, `normalizeInputs`, `continuousQuantileHead`, `flipInvariance`, `inferPositive`, `fixQuantileCrossing` |
| Precision | FP32 by default; optional FP16 or TF32 GEMM inputs |

## How it maps onto jitcast

| Reference | jitcast |
|---|---|
| `TimesFM_2p5_200M_Definition`, `config.json` | `timesfm.TimesFmConfig` (unknown keys and unsupported sizes fail the load) |
| `TimesFM_2p5_200M_torch_module` parameters | `timesfm.TimesFmWeights`: every tensor bound by name and shape |
| `TimesFM_2p5_200M_torch_module.forward` | `timesfm.PatchForward`: `cpu.CpuTimesFm`, `tornado.TimesFmForward` |
| `decode`, `update_running_stats`, `revin` | `timesfm.TimesFmPipeline` |
| `TimesFM_2p5.forecast`, `ForecastConfig` | `timesfm.TimesFmPipeline`, `TimesFmPipeline.Options` |

- **Host and device.** The batch assembly, the input normalization, each patch's running mean and
  deviation, and the post-processing (flip averaging, the continuous quantile head, quantile crossing,
  clamping) run on the host. The tokenizer, the transformer and the heads of each row's last patch run
  on the device, one plan per shape (rows, patches).
- **Decoding without a cache.** The reference decodes with a key-value cache, feeding its median back
  128 steps at a time. jitcast runs every step over the whole sequence instead; attention is causal, so
  the outputs are the same.
- **Kernels.** RoPE with the q and k norms, the causal attention and the last-patch gather are
  TimesFM's own ([KERNELS.md](../KERNELS.md#timesfm-25)); the GEMMs, RMSNorm and the SiLU epilogues are
  shared.

## Options and tuning

`timesfm.*` options are in [CONFIGURATION.md](../CONFIGURATION.md#timesfm-25). The defaults use cuBLAS
FP32 with CUDA graphs on NVIDIA and simdgroup GEMMs on Metal; `-Dtimesfm.tornado.gemm=cublas-fp16` uses
FP16 tensor cores on NVIDIA.

## Limits

- Covariates are not supported (Google's package handles them with a separate linear model); a series
  with covariates fails the call.
- Sequences up to 512 patches (16,384 steps), the model's own limit.
- Validated on `timesfm-2.5-200m`. TimesFM 3.0's weights are under a non-commercial licence and its
  architecture differs; it is not supported.

## Licence

The TimesFM 2.5 weights and Google's inference code are under the Apache License 2.0. jitcast does not
distribute the weights.
