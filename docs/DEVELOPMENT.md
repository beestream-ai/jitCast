# Developing jitrex

[← back to the README](../README.md)

## Architecture

TiRex-2 is an xLSTM model: 12 blocks of `[mLSTM, sLSTM] × 6`, with a variate-mixing attention
block after each time mixer, D = 512, and 82.5M parameters. It gives 9-quantile forecasts for
univariate series, multi-target series, and series with past or future covariates.

jitrex reproduces NXAI's `ForecastModel.forecast()` end to end:
- **Host side, in Java:** batch assembly and padding, the scaler, the postprocessor (trend
  detection, differencing, band calibration) and sign-flip test-time augmentation.
- **Model forward pass:** on a TornadoVM device. It runs as one task graph per batch shape, with
  the activations kept on the device.

Every kernel is a TornadoVM **KernelContext** kernel, so the whole model runs on any TornadoVM
backend: CUDA, OpenCL and Metal. On NVIDIA GPUs the GEMMs can instead use TornadoVM's hybrid
libraries (**cuBLAS / cuBLASLt**) or a **CUDA Tile** kernel written in TornadoVM, and each task
graph can be replayed as a **CUDA graph**. A pure-Java FP32 CPU implementation of the same model
serves as the reference that every device kernel is tested against.

The engine loads NXAI's checkpoints after a one-off conversion to safetensors, and supports the
base model (`NX-AI/TiRex-2`) and the benchmark checkpoints (`-fevbench`, `-gifteval-zs`,
`-gifteval-pretrain`).

## Java layout (`ai.beestream.jitrex`)

| Package | Contents |
|---|---|
| `api` | `TimeSeries`, `Forecast`, `TiRex2Engine` |
| `io` | `SafeTensors` (memory-mapped reader), `Tensor` |
| `model` | `TiRex2Config` (from the embedded model config), `TiRex2Weights` (binds every tensor by name and shape; unknown or missing tensors fail the load) |
| `cpu` | FP32 CPU reference, the oracle for the device kernels: `Ops`, `InputStage`, `Cells` (mLSTM, sLSTM), `CpuReference` (the full forward pass), `CpuTiRex2Engine` |
| `bench` | `TiRexBench` (latency/throughput), `TiRexColdStart`, `TiRexServer` (socket server for NXAI's Python harnesses) |
| `util` | `Json` |
| `host` | `ForecastPipeline` (`ForecastModel.forecast`: 512-series windows, batch assembly and padding, horizon slicing, sign-flip TTA) and `PostProcessor` (trend detection, differencing, band calibration) — shared by every engine |
| `tornado` | `TiRexKernels` (every device kernel, see [KERNELS.md](KERNELS.md)), `PackedWeights` (one device buffer plus per-matrix FP32/FP16 copies for the library GEMMs; fused q\|k\|v\|o\|i\|f, up_gate\|up and the block-diagonal sLSTM gates), `TornadoForward` (a plan per batch shape — rows, known covariates, kept rows — built as one task graph: the input stage, the blocks and the output stage; the GEMM modes; CUDA graphs), `TornadoTiRex2Engine` |

## Scripts (`scripts/`)

| Script | What it does |
|---|---|
| `convert_ckpt.py` | `model.ckpt` + `model-config.yaml` → one `tirex2.safetensors` (FP32, aliases removed, config in the metadata) |
| `spec_forward.py` | An explicit PyTorch spec of `forward()` (basic ops only, no xlstm/flashrnn) checked stage by stage against the goldens; the Java reference is a port of it |
| `dump_golden.py` | Runs the NXAI model on 13 cases and saves the inputs, the batch passed to `forward()`, the activations of every stage, the raw model output and the final forecasts |
| `bench_nxai.py` | NXAI baseline: `workloads` (shared inputs), `run` (latency/throughput, e2e + forward-only), `coldstart` |
| `setup_reference.sh` | The whole reference setup on a CUDA machine, in a work directory: venv, CUDA-kernel check, conversion, CPU + CUDA goldens, NXAI baseline, cold start |
| `tirex_tornado.py`, `run_with_tornado.py` | Run an unmodified NXAI harness with `TiRex2.predict` served by `TiRexServer` |
| `gift_resumable.py` | GIFT-Eval one configuration at a time: results appended as they land, resumable, server restart and retry on failure |
| `eval/` | The evaluation matrix: `eval_nvidia.sh` / `eval_mac.sh` + `summarize_eval.py` / `plot_eval.py` ([PERFORMANCE.md](PERFORMANCE.md)); accuracy parity: `acc_setup.sh` / `acc_run.sh` / `acc_compare.py` ([VALIDATION.md](VALIDATION.md)); `tornado_run.sh` (run a jitrex main class on a TornadoVM SDK), `nxai.sh` (NXAI's benchmark in its venv), `profile.sh` + `kern_sum.py` (nsys), `cublas13_shim.sh` ([INSTALL.md](INSTALL.md#nvidia-extras)), `waitfree.sh` (wait for a shared machine to be free). Each script's header lists its arguments and environment |

```bash
./scripts/setup_reference.sh ~/tirex2-work
```
