# TiRex-2

[← back to the README](../../README.md)

[TiRex-2](https://github.com/NX-AI/tirex-2) is NXAI's zero-shot forecasting model. jitcast reproduces
NXAI's `ForecastModel.forecast()` end to end: 9-quantile forecasts for univariate series,
multi-target series, and series with past or future covariates.

| | |
|---|---|
| Architecture | xLSTM: 12 blocks of `[mLSTM, sLSTM] × 6`, each time mixer followed by attention across variates; D = 512, 82.5M parameters; a residual-block quantile head |
| Checkpoints | `NX-AI/TiRex-2` and `-fevbench` (2,048 + 320 steps = 74 patches); `-gifteval-zs` and `-gifteval-pretrain` (8,192 + 928 = 285 patches, no qk-norm), converted once to safetensors |
| Validated | CPU, CUDA, OpenCL and Metal, against NXAI's `tirex-2` 0.4.0 |
| Engines | `tirex2.tornado.TornadoTiRex2Engine` (TornadoVM), `tirex2.cpu.CpuTiRex2Engine` (FP32 reference) |

## Quick start

jitcast reads one safetensors file per checkpoint. `convert_ckpt.py` downloads a checkpoint from
Hugging Face and writes it with aliases removed and the model config embedded:

```bash
python3.11 -m venv venv && . venv/bin/activate
pip install torch safetensors numpy huggingface_hub pyyaml
python scripts/convert_ckpt.py --ckpt NX-AI/TiRex-2 --out tirex2.safetensors
```

```java
try (ForecastEngine engine = TornadoTiRex2Engine.load(Path.of("tirex2.safetensors"))) {
    List<Forecast> forecasts = engine.forecast(List.of(TimeSeries.univariate(history)), 64);
    float[] median = forecasts.get(0).median(0);        // values[target][quantile][step]
}
```

## Features

| Feature | Supported |
|---|---|
| Inputs | univariate, multi-target, past and future-known covariates, mixed batches, NaN gaps, binary series, contexts shorter or longer than the model's |
| Horizon | up to the checkpoint's future length (320 or 928 steps); longer horizons are cut, as in NXAI's engine |
| Test-time augmentation | sign flip and trend differencing, defaulting to the checkpoint's settings |
| Precision | FP32 by default; optional FP16 GEMM inputs (FP32 accumulation and residual stream) and FP16 sLSTM recurrent weights |
| Harness | `bench.TiRexServer` serves NXAI's Python harnesses over a local socket |

## How it maps onto jitcast

- **Host** (`tirex2.ForecastPipeline`, `PostProcessor`): batch assembly and padding, trend detection,
  differencing, band calibration, sign-flip TTA.
- **Device** (`tirex2.tornado.TornadoForward`): one plan per batch shape — scaler and patching, mLSTM
  and sLSTM cells, head norms, variate attention, residual heads — with activations kept on the
  device. Kernels in [KERNELS.md](../KERNELS.md#tirex-2).
- **Weights** (`tirex2.TiRex2Weights`, `tirex2.tornado.PackedWeights`): bound by name and shape;
  q|k|v|o|i|f, the FFN's up-gate|up and the block-diagonal sLSTM gates are fused at load.
- **Reference** (`tirex2.cpu.CpuReference`): a port of `scripts/spec_forward.py`, an executable spec
  of NXAI's `forward()` in basic PyTorch ops.

## Options and tuning

`tirex.tornado.*` options (GEMM mode, sLSTM and mLSTM kernels, CUDA graphs, fused epilogues) and the
per-platform flags are in [CONFIGURATION.md](../CONFIGURATION.md#tirex-2). In short: on NVIDIA,
`-Dtirex.tornado.gemm=cublas-auto` is fastest; on Apple Silicon, `half`.

## Limits

- sLSTM heads must be 128 wide (every published checkpoint is).

## Licence

The TiRex-2 model and weights are NX-AI's, under their own licence terms. jitcast does not
distribute the weights.
