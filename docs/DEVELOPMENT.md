# Developing jitcast

[← back to the README](../README.md)

## Architecture

Each model family reproduces its reference engine's `forecast()` end to end, with one structure:

- **API** (`api`): a `ForecastEngine` forecasts a batch of `TimeSeries` (targets, past covariates,
  known-future covariates; NaN marks a missing value) into one `Forecast` of quantiles per series.
- **Host pipeline, in Java**: batch assembly and padding, scaling, decoding, calibration and
  test-time augmentation, over a forward-pass interface the family defines. Splitting a call into
  passes (`host.SeriesWindows`) and sign-flip averaging (`host.SignFlip`) are shared.
- **Forward pass on a TornadoVM device**: one plan per batch shape, compiled once and kept in a
  `tornado.PlanCache` (least recently used evicted within a device-memory budget), activations on the
  device. GEMMs, RMSNorm, SwiGLU and their epilogues are shared kernels (`tornado.Kernels`), selected
  by `tornado.GemmMode` for Timestral and Chronos-2 and by `TornadoForward.GemmMode` for TiRex-2.
  The rest are the family's own kernels.
- **CPU reference**: a plain-Java FP32 forward pass from the shared operators (`cpu.Ops`), the
  oracle the device kernels are tested against.

Every kernel is a TornadoVM `KernelContext` kernel (no `@Parallel` loops) with an explicit worker
grid and bounds checks, so one code base runs on CUDA, OpenCL and Metal.

## Java layout (`ai.beestream.jitcast`)

| Package | Contents |
|---|---|
| `api` | `TimeSeries`, `Forecast`, `ForecastEngine` |
| `io` | `SafeTensors` (memory-mapped reader), `Tensor` |
| `host` | `SeriesWindows` (splits a call by series count and token budget), `SignFlip` |
| `cpu` | `Ops`: shared FP32 operators (linear, RMSNorm, SiLU, ReLU, sigmoid) |
| `tornado` | `Devices` (device, memory budget, TornadoVM defaults), `PlanCache`, `Kernels` (shared GEMMs, norms, epilogues; see [KERNELS.md](KERNELS.md)), `GemmMode` |
| `util` | `Json`, `Options` (strict system-property readers) |
| `bench` | `TiRexBench`, `TiRexColdStart`, `TiRexServer`, `PlanMemoryProbe`, `TimestralBench`, `TimestralServer`, `Chronos2Bench`, `Wire` (the servers' protocol) |
| `tirex2` | `TiRex2Config`, `TiRex2Weights`, `ForecastPipeline`, `PostProcessor` (trend, differencing, band calibration), `ModelBatch`, `ForwardPass` |
| `tirex2.cpu` | FP32 reference: `TiRexOps`, `InputStage`, `Cells` (mLSTM, sLSTM), `CpuReference`, `CpuWeights`, `CpuTiRex2Engine` |
| `tirex2.tornado` | `TiRexKernels`, `PackedWeights` (fused projections, FP16 copies), `TornadoForward` (plans, GEMM modes, CUDA graphs), `TornadoTiRex2Engine` |
| `timestral` | `TimestralConfig`, `TimestralWeights`, `BlockInput` / `BlockForward`, `BlockLayout`, `Scaler`, `Quantiles`, `TargetEncoding`, `BlockPrep`, `TimestralPipeline`, `Timing` |
| `timestral.cpu` | `CpuTimestral`, `CpuTimestralEngine` |
| `timestral.tornado` | `TimestralKernels`, `TimestralPacked`, `TimestralForward`, `TornadoTimestralEngine` |
| `chronos2` | `Chronos2Config`, `Chronos2Weights`, `ModelInput` / `ModelForward`, `ModelLayout`, `ModelPrep`, `QuantileMath`, `Chronos2Pipeline` |
| `chronos2.cpu` | `CpuChronos2`, `CpuChronos2Engine` |
| `chronos2.tornado` | `Chronos2Kernels`, `Chronos2Packed`, `Chronos2Forward` (rows regrouped for group attention), `TornadoChronos2Engine` |

## Adding a model family

The Timestral port is the template. Validate each step before the next:

1. **Read the reference end to end**: config, weights, every layer, masks, the forecaster loop. Note
   what runs in float64 or depends on the device. Check the weights' licence.
2. **Golden script** (`scripts/<family>/dump_golden.py`): run the reference on cases covering its
   inputs (univariate, multivariate, NaNs, ragged batches, covariates, long contexts and horizons) and
   record each forward call's inputs, every stage, the output, the forecasts, and each stage's
   distance from a float64 replay (the reference's noise floor).
3. **Config and weights**: unsupported settings, inconsistent sizes and unbound tensors fail the load.
4. **CPU reference and pipeline** against the goldens: every stage within 1e-4 (or 4× the noise
   floor), forward-call inputs bit for bit, forecasts within 1e-4.
5. **TornadoVM engine**: shared kernels first, then the family's own; a device test against the
   goldens and a large mixed batch against the CPU reference, on Metal, OpenCL and CUDA.
6. **Docs**: a page in `docs/models/`, and the family's sections in the shared docs. Benchmarks go to
   [jitCast-benchmarking](https://github.com/beestream-ai/jitCast-benchmarking).

## Tests

`./mvnw test` runs the CPU tests; `-Paccel-tests` adds the device tests (`*AccelTest`). Golden tests
read their paths from system properties and are skipped without them; see [USAGE.md](USAGE.md#tests).
CI builds and runs the CPU tests on every push.

## Scripts (`scripts/`)

| Script | What it does |
|---|---|
| `convert_ckpt.py` | TiRex-2 `model.ckpt` + `model-config.yaml` → one safetensors file (FP32, aliases removed, config embedded) |
| `spec_forward.py` | An explicit PyTorch spec of TiRex-2's `forward()` in basic ops, checked against the goldens; the Java reference ports it |
| `dump_golden.py` | TiRex-2 goldens: inputs, the batch passed to `forward()`, every stage, raw output, forecasts |
| `setup_reference.sh <dir>` | The whole TiRex-2 reference setup on a CUDA machine: venv, conversion, CPU and CUDA goldens |
| `timestral/dump_golden.py` | Timestral goldens: every `predict_block` call's arguments, stages and output, forecasts, float64 distances |
| `chronos2/dump_golden.py` | Chronos-2 goldens: every forward call's arguments, stages and output, forecasts, float64 distances |
