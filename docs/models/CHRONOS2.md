# Chronos-2

[← back to the README](../../README.md)

[Chronos-2](https://huggingface.co/amazon/chronos-2) is Amazon's time-series foundation model.
jitcast reproduces `Chronos2Pipeline.predict()` from Amazon's
[`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting) package end to end:
quantile forecasts for univariate and multivariate series, with past-only and known-future
covariates, cross-learning across a batch, and any horizon (past one forward pass, by the
reference's long-horizon unrolling).

| | |
|---|---|
| Architecture | T5-style encoder over 16-step patches: time attention with RoPE along each series, group attention across the series of a group at each patch, ReLU feed-forward layers, RMSNorm; a `[REG]` token and one token per forecast patch; residual-MLP patch embedding and quantile head |
| Checkpoints | `amazon/chronos-2` (120M, 21 quantiles) and `autogluon/chronos-2-small` (28M, 13 quantiles) on Hugging Face, loaded as published (`config.json` + `model.safetensors`) |
| Validated | both checkpoints on CPU, Metal, OpenCL and CUDA ([VALIDATION.md](../VALIDATION.md#chronos-2)) |
| Performance | [PERFORMANCE.md](../PERFORMANCE.md#chronos-2) |
| Engines | `chronos2.tornado.TornadoChronos2Engine` (TornadoVM), `chronos2.cpu.CpuChronos2Engine` (FP32 reference) |

## Quick start

```bash
# a checkpoint (Apache 2.0)
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

The pipeline's settings (context cap, cross-learning, patches per forward pass, the unrolled levels)
are `Chronos2Pipeline.Options`, with the reference's defaults:

```java
Chronos2Pipeline p = engine.pipeline().withOptions(
        Chronos2Pipeline.Options.defaults(engine.config()).withCrossLearning(true));
```

## Features

| Feature | Supported |
|---|---|
| Inputs | univariate series, several jointly forecast targets, past-only and known-future numeric covariates, mixed batches, NaN gaps and late starts, series of different lengths in one batch |
| Cross-learning | one group per batch (every series attends every other), as the reference's `cross_learning=True`; batches of 256 rows by default |
| Context | up to the checkpoint's 8,192 steps (longer contexts keep the most recent) |
| Horizon | any: up to 64 patches (1,024 steps) per forward pass, then the reference's long-horizon unrolling |
| Precision | FP32 throughout by default; optional FP16 or TF32 GEMM inputs |

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

- **The forward pass.** Each row (a target or a covariate of a series) has one token per 16-step
  context patch, the `[REG]` token and one token per forecast patch. Each encoder block runs time
  attention along a row (RoPE, bidirectional, over the tokens with an observation), group attention
  across the rows of a group at each token (the targets and covariates of one series, or the whole
  batch with cross-learning), and a ReLU feed-forward layer. As in the reference, attention scores are
  not scaled by `1/√d_kv`.
- **Host and device.** The instance norm (a row's mean and standard deviation, then arcsinh), the
  patch tokens (time encoding, scaled values, observation mask) and the inverse norm run on the host,
  in `chronos2.ModelPrep`, with the rows in parallel. The patch embedding, the encoder and the output
  patch embedding run on the device as one task graph per shape (rows, context patches, output
  patches). The device engine reorders the rows so that each group is a contiguous run, which the
  group-attention kernel needs, and restores the order after.
- **Long horizons.** Past `max_output_patches · 16` steps (1,024) the pipeline unrolls as the
  reference does: every row is repeated once per unrolled level (0.1 … 0.9), each copy's context is
  extended with that level of the forecast so far (and the known covariates with their real future),
  and the next pass's forecasts of all copies are merged back into the quantile levels as a weighted
  sample.
- **Token budget.** A forward pass larger than `-Dchronos2.maxTokens` tokens runs as several, split
  between groups, with the same forecasts.
- **Kernels.** Token assembly, RoPE, the time and group attention and the future-token gather are
  Chronos-2's own ([KERNELS.md](../KERNELS.md#chronos-2)); the GEMMs (with a ReLU epilogue) and RMSNorm
  are shared.

## Options

| Property | Default | Effect |
|---|---|---|
| `jitcast.tornado.device` | TornadoVM's default | The device, `<backend>:<device>` |
| `chronos2.tornado.gemm` | `auto` | As `timestral.tornado.gemm` ([CONFIGURATION.md](../CONFIGURATION.md#chronos-2)): `cublas` on CUDA, `simdgroup` on Metal when Metal is TornadoVM's default backend, `blocked` elsewhere |
| `chronos2.tornado.blockedMinRows` | `128` | The fewest GEMM rows that use the register-blocked kernel |
| `chronos2.tornado.wideMinRows` | `1024` | The fewest GEMM rows that use the register-blocked kernel with 64 × 128 tiles |
| `chronos2.tornado.simdgroupWideMinRows` | `64` | The fewest GEMM rows that use the 64 × 64 simdgroup kernel (the 32 × 32 one below) |
| `chronos2.tornado.splitKTarget`, `splitKMax` | `320`, `16` | Split K for simdgroup GEMMs with few tiles, as for Timestral |
| `chronos2.tornado.cudaGraph` | `true` with cuBLAS | Capture the task graph once and replay it as a CUDA graph |
| `chronos2.tornado.plans` | `8` | Shapes kept compiled (least recently used are evicted) |
| `chronos2.tornado.profile` | `false` | Print TornadoVM's per-task kernel times after every execution |
| `chronos2.maxTokens` | `131072` | The most tokens one forward pass takes; forecasts do not depend on the split |

## Limits

- Categorical covariates (target-encoded in the reference) are not supported: a series with
  categorical covariates fails the call; encode them as numbers first.
- Only float32 checkpoints with the ReLU feed-forward layer, non-overlapping patches and equal input
  and output patch sizes load; every published Chronos-2 checkpoint is one. On the device, heads must
  be 16 to 64 wide, in multiples of 16 (all are 64).
- Validated on `chronos-2` and `chronos-2-small`. `autogluon/chronos-2-synth` shares their
  architecture and loads with the same code, but has no goldens.

## Licence

The Chronos-2 weights and Amazon's inference code are under the Apache License 2.0. jitcast does not
distribute the weights.
