# Developing jitcast

[← back to the README](../README.md)

## Architecture

jitcast runs zero-shot time-series forecasting models. Each model family reproduces its reference
engine's `forecast()` end to end, and shares one engine structure with the others:

- **API** (`api`): a `ForecastEngine` forecasts a batch of `TimeSeries` (targets, past covariates,
  known-future covariates; NaN marks a missing value) and returns one `Forecast` of quantiles per
  series. Every engine of every family implements it.
- **Host side, in Java**: the family's forecast pipeline (batch assembly and padding, scaling,
  decoding, calibration, test-time augmentation), over a forward-pass interface it defines. The
  splitting of a call into forward passes (`host.SeriesWindows`) and sign-flip averaging
  (`host.SignFlip`) are shared.
- **Forward pass on a TornadoVM device**: one task graph per batch shape, compiled once and cached,
  with the activations kept on the device. The GEMMs, RMSNorm, SwiGLU and their epilogues are shared
  kernels (`tornado.Kernels`), run in a `tornado.GemmMode`: TornadoVM's own FP32 or FP16 kernels on any
  backend, Apple's simdgroup matrices on Metal, or cuBLAS on CUDA, with CUDA graphs. The rest are the family's own kernels.
- **CPU reference**: a plain-Java FP32 port of the forward pass, from the shared operators
  (`cpu.Ops`), which the device kernels are tested against.

Every kernel is a TornadoVM **KernelContext** kernel (no `@Parallel` loops), launched with an
explicit worker grid, so the same code runs on CUDA, OpenCL and Metal.

## Java layout (`ai.beestream.jitcast`)

| Package | Contents |
|---|---|
| `api` | `TimeSeries`, `Forecast`, `ForecastEngine` (implemented by every engine of every family) |
| `io` | `SafeTensors` (memory-mapped reader), `Tensor` |
| `host` | Model-agnostic host pieces: `SeriesWindows` (splits a forecast call into forward passes by series count and token budget) and `SignFlip` (sign-flip test-time augmentation) |
| `cpu` | `Ops`: the FP32 operators shared by every family's CPU reference (linear, RMSNorm, SiLU, ReLU, sigmoid), the oracles of the shared kernels |
| `tornado` | `Kernels`: the device kernels shared by every family — the FP32 (tiled, register-blocked, simdgroup), FP16 and CUDA Tile GEMMs with SiLU or ReLU epilogues, RMSNorm, SwiGLU and the epilogues around a cuBLAS GEMM (see [KERNELS.md](KERNELS.md)); `GemmMode`: how a family's GEMMs run (TornadoVM kernels or cuBLAS) |
| `util` | `Json` |
| `bench` | `TiRexBench` (latency/throughput), `TiRexColdStart`, `TiRexServer` (socket server for NXAI's Python harnesses), `PlanMemoryProbe`, `TimestralBench`, `TimestralServer`, `Chronos2Bench` |
| `tirex2` | `TiRex2Config` (from the embedded model config), `TiRex2Weights` (binds every tensor by name and shape; unknown or missing tensors fail the load), `ForecastPipeline` (`ForecastModel.forecast`: batch assembly and padding, horizon slicing, calibration) and `PostProcessor` (trend detection, differencing, band calibration), `ModelBatch` (the batch `forward()` receives) and `ForwardPass` (the forward-pass interface) — shared by both TiRex-2 engines |
| `tirex2.cpu` | FP32 CPU reference, the oracle for the device kernels: `TiRexOps` (head LayerNorm, gate functions, residual block, SwiGLU FFN), `InputStage`, `Cells` (mLSTM, sLSTM), `CpuReference` (the full forward pass), `CpuWeights` (heap copies of the weights it reads), `CpuTiRex2Engine` |
| `tirex2.tornado` | `TiRexKernels` (the TiRex-2-specific device kernels: scaler, mLSTM and sLSTM cells, head norms, variate attention, output stage), `PackedWeights` (one device buffer plus per-matrix FP32/FP16 copies for the library GEMMs; fused q\|k\|v\|o\|i\|f, up_gate\|up and the block-diagonal sLSTM gates), `TornadoForward` (a plan per batch shape — rows, known covariates, kept rows — built as one task graph: the input stage, the blocks and the output stage; the GEMM modes; CUDA graphs), `TornadoTiRex2Engine` |
| `timestral` | `TimestralConfig` (`config.json`), `TimestralWeights` (binds every tensor; rearranges q\|k\|v and the MLP input), `BlockInput` / `BlockForward` (one `predict_block`), `BlockLayout` (token layout and masks), `Scaler`, `Quantiles`, `TargetEncoding` (categorical covariates), `BlockPrep` (the host ends of a block), `TimestralPipeline` (`TimestralForecaster.forecast`), `Timing` (opt-in phase timing) — see [models/TIMESTRAL.md](models/TIMESTRAL.md) |
| `timestral.cpu` | `CpuTimestral` (the FP32 reference of `predict_block`), `CpuTimestralEngine` |
| `timestral.tornado` | `TimestralKernels` (token assembly, RoPE, time and variate attention, gate, decoder gather), `TimestralPacked`, `TimestralForward` (a task graph per block shape), `TornadoTimestralEngine` |
| `chronos2` | `Chronos2Config` (`config.json`), `Chronos2Weights` (binds every tensor; fuses q\|k\|v), `ModelInput` / `ModelForward` (one `Chronos2Model.forward`), `ModelLayout` (token layout, masks, groups), `ModelPrep` (instance norm, patch tokens, inverse norm), `QuantileMath` (quantile interpolation and weighted quantiles), `Chronos2Pipeline` (`Chronos2Pipeline.predict`: batching, long-horizon unrolling) — see [models/CHRONOS2.md](models/CHRONOS2.md) |
| `chronos2.cpu` | `CpuChronos2` (the FP32 reference of the forward pass), `CpuChronos2Engine` |
| `chronos2.tornado` | `Chronos2Kernels` (token assembly, RoPE, time and group attention, future-token gather), `Chronos2Packed`, `Chronos2Forward` (a task graph per shape; rows regrouped for the group attention), `TornadoChronos2Engine` |

## Adding a model family

The Timestral port is the template; it took these steps, each validated before the next:

1. **Read the reference end to end** (config, weights, every layer, the masks, the forecaster loop),
   and note what runs in float64 or depends on the device. Check the weights' licence.
2. **A golden script** (`scripts/<family>/dump_golden.py`) that runs the reference on cases covering
   its inputs (univariate, multivariate, NaNs, ragged batches, covariates, long contexts and
   horizons), recording each forward call's inputs, every stage, its output, the forecasts, and each
   stage's distance from a float64 replay (the reference's own noise floor).
3. **Config and weights** (`<family>.<Family>Config`, `<Family>Weights`): unknown config keys,
   unsupported settings and unbound tensors fail the load.
4. **CPU reference and pipeline** against the goldens: every stage within 1e-4 (or 4× the noise floor),
   every forward call's inputs bit for bit, the forecasts within 1e-4.
5. **TornadoVM engine**: the shared kernels first, then the family's own; a device test against the
   goldens and a large mixed batch against the CPU reference; on Metal, OpenCL and CUDA.
6. **Benchmark** against the reference on the same inputs (a workload file both sides read), then
   documentation: a page in `docs/models/`, and the shared docs' family sections.

## Tests

`./mvnw test` runs the CPU tests; `-Paccel-tests` adds the device tests (`*AccelTest`). Each family's
golden tests read their paths from system properties and are skipped when those are not set; see
[USAGE.md](USAGE.md#tests).

## Scripts (`scripts/`)

| Script | What it does |
|---|---|
| **TiRex-2** | |
| `convert_ckpt.py` | `model.ckpt` + `model-config.yaml` → one `tirex2.safetensors` (FP32, aliases removed, config in the metadata) |
| `spec_forward.py` | An explicit PyTorch spec of `forward()` (basic ops only, no mLSTM or FlashRNN kernels) checked stage by stage against the goldens; the Java reference is a port of it |
| `dump_golden.py` | Runs the NXAI model on 13 cases and saves the inputs, the batch passed to `forward()`, the activations of every stage, the raw model output and the final forecasts |
| `bench_nxai.py` | NXAI baseline: `workloads` (shared inputs), `run` (latency/throughput, e2e + forward-only), `coldstart` |
| `setup_reference.sh` | The whole reference setup on a CUDA machine, in a work directory: venv, CUDA-kernel check, conversion, CPU + CUDA goldens, NXAI baseline, cold start |
| `tirex_tornado.py`, `run_with_tornado.py` | Run an unmodified NXAI harness with `TiRex2.predict` served by `TiRexServer` |
| `gift_resumable.py` | GIFT-Eval one configuration at a time: results appended as they land, resumable, server restart and retry on failure |
| **Timestral** | |
| `timestral/dump_golden.py` | Runs Mistral's Timestral forecaster on 13 cases and saves every `predict_block` call's arguments, stages and output, the forecasts, and each stage's distance from a float64 replay |
| `timestral/bench_reference.py` | The Timestral benchmark workloads, and the reference's latency on them (CPU, CUDA, or MPS with its float64 steps on the CPU) |
| `timestral/eval_mac.sh` | The Apple-silicon matrix: the reference on CPU and MPS against jitcast on Metal and OpenCL, per checkpoint |
| `timestral/fev_eval.py` | fev-bench for both engines under one harness: the reference in process, jitcast through `TimestralServer`; resumable, one result row per task |
| **Chronos-2** | |
| `chronos2/dump_golden.py` | Runs Amazon's Chronos-2 pipeline on 13 cases and saves every forward call's arguments, stages and output, the forecasts, and each stage's distance from a float64 replay |
| `chronos2/bench_reference.py` | The Chronos-2 benchmark workloads, and the reference's latency on them (CPU, CUDA or MPS) |
| **Evaluation** | |
| `eval/` | The evaluation matrix: `eval_nvidia.sh` / `eval_mac.sh` + `summarize_eval.py` / `plot_eval.py` ([PERFORMANCE.md](PERFORMANCE.md)); accuracy parity: `acc_setup.sh` / `acc_run.sh` / `acc_compare.py` ([VALIDATION.md](VALIDATION.md)); `tornado_run.sh` (run a jitcast main class on a TornadoVM SDK), `nxai.sh` (NXAI's benchmark in its venv), `profile.sh` + `kern_sum.py` (nsys), `cublas13_shim.sh` ([INSTALL.md](INSTALL.md#nvidia-extras)), `waitfree.sh` (wait for a shared machine to be free). Each script's header lists its arguments and environment |

The whole TiRex-2 reference setup on a CUDA machine (venv, conversion, goldens, NXAI baseline):

```bash
./scripts/setup_reference.sh ~/tirex2-work
```
