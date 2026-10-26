# TiRex-2

[← back to the README](../../README.md)

[TiRex-2](https://github.com/NX-AI/tirex-2) is NXAI's zero-shot forecasting model. jitcast reproduces
NXAI's `ForecastModel.forecast()` end to end: 9-quantile forecasts for univariate series,
multi-target series, and series with past or future covariates.

| | |
|---|---|
| Architecture | xLSTM: 12 blocks of `[mLSTM, sLSTM] × 6`, each time mixer followed by attention across variates; D = 512, 82.5M parameters; a residual-block quantile head |
| Checkpoints | `NX-AI/TiRex-2` and `-fevbench` (2,048 + 320 steps = 74 patches); `-gifteval-zs` and `-gifteval-pretrain` (8,192 + 928 = 285 patches, no qk-norm), converted once to safetensors |
| Validated | CPU, CUDA, OpenCL and Metal, against NXAI's tirex-2 0.4.0; benchmark parity with NXAI on fev-bench and GIFT-Eval ([VALIDATION.md](../VALIDATION.md#tirex-2)) |
| Performance | [PERFORMANCE.md](../PERFORMANCE.md#tirex-2) |
| Engines | `tirex2.tornado.TornadoTiRex2Engine` (TornadoVM), `tirex2.cpu.CpuTiRex2Engine` (FP32 reference) |

## Quick start

jitcast reads one safetensors file per checkpoint. `convert_ckpt.py` downloads a checkpoint from
Hugging Face and writes that file, with aliases removed and the model config embedded:

```bash
python3.11 -m venv venv && . venv/bin/activate        # Python 3.11 or newer
pip install torch safetensors numpy huggingface_hub pyyaml
python scripts/convert_ckpt.py --ckpt NX-AI/TiRex-2 --out tirex2.safetensors
python scripts/convert_ckpt.py --ckpt NX-AI/TiRex-2-gifteval-zs --out TiRex-2-gifteval-zs.safetensors
```

```java
try (ForecastEngine engine = TornadoTiRex2Engine.load(Path.of("tirex2.safetensors"))) {
    List<Forecast> forecasts = engine.forecast(List.of(TimeSeries.univariate(history)), 64);
    float[] median = forecasts.get(0).median(0);        // values[target][quantile][step]
}
```

`ForecastPipeline(config, forward, batchSize, ttaDiff, ttaSignFlip)` sets the series per forward
pass (512) and the trend differencing and sign flip, which default to the checkpoint's values.

## Features

| Feature | Supported |
|---|---|
| Inputs | univariate, multi-target, past covariates, future-known covariates, mixed batches, NaN gaps, binary series, contexts shorter or longer than the model context |
| Horizon | up to the checkpoint's future length (320 or 928 steps); longer horizons are cut to it, as in NXAI's engine |
| Test-time augmentation | sign-flip TTA and trend differencing, defaulting to the checkpoint's settings |
| Precision | FP32 throughout; optional FP16 GEMM inputs (FP32 accumulation and residual stream); optional FP16 sLSTM recurrent weights |
| Harness | `bench.TiRexServer` serves `TiRex2.predict` to NXAI's own Python harnesses over a local socket |

## How it maps onto jitcast

- **Host side** (`tirex2.ForecastPipeline`, `PostProcessor`): batch assembly and padding, the trend
  detection, differencing and band calibration, sign-flip TTA.
- **Device side** (`tirex2.tornado.TornadoForward`): one task graph per batch shape — the scaler and
  patching, the mLSTM and sLSTM cells, the head norms, variate attention, the residual heads — with
  the activations kept on the device. Its kernels are in [KERNELS.md](../KERNELS.md#tirex-2).
- **Weights** (`tirex2.TiRex2Weights`, `tirex2.tornado.PackedWeights`): bound by name and shape; the q|k|v|o|i|f
  projections, the FFN's up-gate|up and the block-diagonal sLSTM gates are fused at load time.
- **Reference** (`tirex2.cpu.CpuReference`): a port of `scripts/spec_forward.py`, an executable spec of
  NXAI's `forward()` in basic PyTorch ops.

## Options

TiRex-2's engine options (`tirex.tornado.*`: GEMM mode, sLSTM and mLSTM kernels, CUDA graphs, fused
epilogues) and the per-platform recommendations are in [CONFIGURATION.md](../CONFIGURATION.md#tirex-2).

## NXAI's engine on a newer CUDA toolkit than the driver

NXAI's TiRex-2 runs its sLSTM layers through FlashRNN, which compiles its CUDA kernels on first use
with the toolkit `CUDA_HOME` points to, and only as PTX (`-gencode arch=compute_80,code=compute_80`
in `flashrnn/cuda_init.py`). The driver compiles that PTX at load time, and a driver cannot compile
PTX from a newer toolkit than its own: with CUDA 13.4 on a driver that supports CUDA 13.3, every NXAI
forecast fails with `RuntimeError: Errors during CUDA kernel calls forward` (CUDA error 222, "the
provided PTX was compiled with an unsupported toolchain"). This matters only for the NXAI baselines.
Point FlashRNN at a toolkit no newer than the driver, for example the one PyTorch was built for, and
clear its build cache:

```bash
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
rm -rf ~/.cache/torch_extensions/py*_cu*
```

## Licence

The TiRex-2 model and its weights are NX-AI's, under their own licence terms. jitcast does not
distribute the weights.
