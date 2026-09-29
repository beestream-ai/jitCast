# jitrex

**jitrex** is an inference engine for [TiRex-2](https://github.com/NX-AI/tirex-2), NXAI's zero-shot
time-series forecasting model, written in Java on [TornadoVM](https://github.com/beehive-lab/TornadoVM).

## 1. What jitrex is

TiRex-2 is an xLSTM model: 12 blocks of `[mLSTM, sLSTM] × 6`, with a variate-mixing attention
block after each time mixer, D = 512, and 82.5M parameters. It gives 9-quantile forecasts for
univariate series, multi-target series, and series with past or future covariates.

jitrex reproduces NXAI's `ForecastModel.forecast()` end to end:
- **Host side, in Java:** batch assembly and padding, the scaler, the postprocessor (trend
  detection, differencing, band calibration) and sign-flip test-time augmentation.
- **Model forward pass:** on a TornadoVM device. It runs as a chain of task graphs, one per
  block, with the activations kept on the device.

Every kernel is a TornadoVM **KernelContext** kernel, so the whole model runs on any TornadoVM
backend: CUDA, OpenCL and Metal. On NVIDIA GPUs the GEMMs can instead use TornadoVM's hybrid
libraries (**cuBLAS / cuBLASLt**) or a **CUDA Tile** kernel written in TornadoVM, and each task
graph can be replayed as a **CUDA graph**. A pure-Java FP32 CPU implementation of the same model
serves as the reference that every device kernel is tested against.

The engine loads NXAI's checkpoints after a one-off conversion to safetensors, and supports the
base model (`NX-AI/TiRex-2`) and the benchmark checkpoints (`-fevbench`, `-gifteval-zs`,
`-gifteval-pretrain`).

On an RTX 5070 Ti, jitrex is **2.0–4.1× faster** than NXAI's PyTorch engine end to end, and
starts **5.8× faster** (section 7). On GIFT-Eval and fev-bench it gives the same scores as NXAI's
FP32 reference to about 1e-6 (section 4).

## 2. Supported configurations

### Hardware and backends

| Hardware | TornadoVM backend | GEMM modes | Status |
|---|---|---|---|
| NVIDIA GPU | CUDA | `tiled`, `half`, `cublas`, `cublas-tf32`, `cublas-fp16`, `cublas-auto`, `cutile` | **Validated and benchmarked** on an RTX 5070 Ti (sm_120, Blackwell), driver 580, CUDA 13.0 |
| Apple Silicon GPU | Metal | `tiled`, `half` | **Validated** on an M4 Pro |
| Apple Silicon GPU | OpenCL | `tiled`, `half` | **Validated** on an M4 Pro |
| NVIDIA GPU | OpenCL | `tiled`, `half` | Expected to work (portable kernels), not tested |
| Intel / AMD GPUs, OpenCL CPU runtimes | OpenCL | `tiled`, `half` | Expected to work, not tested |
| Any CPU | none (plain JVM) | — | `CpuTiRex2Engine`: the FP32 reference, for validation and not for speed |

Some kernels adapt to the device's maximum work-group size (for example 256 on OpenCL on macOS, 1024 on Metal and CUDA). The fused sLSTM norm kernel needs 512 threads and is skipped on smaller devices.

### Hybrid APIs and CUDA features

| API | Used for | How to enable | Requirements |
|---|---|---|---|
| cuBLAS: `cublasSgemm`, `cublasSgemmTF32` | FP32 / TF32 GEMMs | `-Dtirex.tornado.gemm=cublas` / `cublas-tf32` | CUDA backend; `tornado-cublas` in the SDK |
| cuBLAS: `cublasGemmEx` (FP16 in, FP32 or FP16 out) | FP16 tensor-core GEMMs | `cublas-fp16`, `cublas-auto` | cuBLAS 13 on Blackwell (see *Installation*) |
| cuBLASLt: `ltMatmulFP16` | FP16 GEMMs | `-Dtirex.tornado.fp16Kernel=lt-h16` | as above |
| CUDA Tile (`TileContext`) | FP16 GEMM on tensor cores, written in TornadoVM | `-Dtirex.tornado.gemm=cutile` | CUDA Toolkit 13.3+ (`tileiras`), compute capability 8.0+, driver R580+ |
| CUDA graphs | replaying each task graph | `-Dtirex.tornado.cudaGraph=true` (default with `cublas*`, `cutile`) | CUDA backend |
| cuDNN | — | not used | — |

### Model features

| Feature | Supported |
|---|---|
| Checkpoints | `NX-AI/TiRex-2`, `-fevbench` (context 2048 + future 320 = 74 patches); `-gifteval-zs`, `-gifteval-pretrain` (8192 + 928 = 285 patches, no qk-norm) |
| Inputs | univariate, multi-target, past covariates, future-known covariates, mixed batches, NaN gaps, binary series, contexts shorter or longer than the model context |
| Test-time augmentation | sign-flip TTA and trend differencing, defaulting to the checkpoint's settings |
| Precision | FP32 throughout; optional FP16 GEMM inputs (FP32 accumulation and residual stream); optional FP16 sLSTM recurrent weights |

## 3. Installation

### Requirements
- **JDK 22 or newer.** It is tested with JDK 25. It builds with `--release 22` and uses TornadoVM's `jdk22plus` SDK family.
- **A TornadoVM SDK**, version 7.0.1 or newer, for the backend you want.
- **Python 3.11+ with PyTorch**, once, to convert NXAI's checkpoints. It is also needed for the NXAI baselines and the accuracy benchmarks.

### 3.1 Install TornadoVM

Prebuilt SDKs for OpenCL, CUDA, Metal, or all of them are at [tornadovm.org/downloads](https://www.tornadovm.org/downloads) and on SDKMAN!. Pick the `jdk22plus` family:

```bash
sdk list tornadovm                              # the available <version>-<jdk>-<backend> combinations
sdk install tornadovm 7.0.1-jdk22plus-metal     # macOS (Apple Silicon); or ...-opencl
sdk install tornadovm 7.0.1-jdk22plus-cuda      # NVIDIA
export TORNADOVM_HOME=<the SDK directory>       # SDKMAN! sets this for you
tornado --devices                               # lists the backends and devices, numbered <backend>:<device>
```

`tornado --generate-argfile` regenerates `$TORNADOVM_HOME/tornado-argfile` (the JVM flags jitrex is launched with) after switching JDK. To build TornadoVM from source, for example for its `develop` branch, see the [TornadoVM installation guide](https://tornadovm.readthedocs.io/en/latest/installation.html).

**NVIDIA extras**
- **cuBLAS 13 on Blackwell.** TornadoVM's cuBLAS provider loads `libcublas.so.12` first. If a CUDA 12.0 cuBLAS is installed next to CUDA 13, it silently runs generic, non-tensor-core kernels on sm_120. Point `LD_LIBRARY_PATH` at a directory where `lib{cublas,cublasLt,cudart}.so.12` link to the CUDA 13 libraries; `scripts/cyclone/exp_shim.sh` builds such a shim.
- **CUDA Tile (`cutile`)** needs toolkit 13.3+. A user-space install is enough: `pip install 'cuda-tile[tileiras]'` into a venv, then pass `-Dtornado.cuda.nvcc=<venv>/lib/python3.*/site-packages/nvidia/cu13/bin/nvcc`. The first run compiles each GEMM task with `tileiras`, which takes 2–3 minutes per plan layout; after that the CUDA code cache serves it.

The CUDA results in this README were measured on a TornadoVM 7.0.2 development build (`develop` at a86db1ace).

### 3.2 Build jitrex

```bash
git clone https://github.com/beestream-ai/jitcast && cd jitrex
./mvnw -q package -DskipTests                                     # TornadoVM 7.0.1 API from Maven Central
./mvnw -q package -DskipTests -Dtornadovm.version=7.0.2-jdk22plus-dev   # or another installed version
```

### 3.3 Get the weights

jitrex reads one safetensors file per checkpoint. `convert_ckpt.py` downloads a checkpoint from Hugging Face and writes that file, with aliases removed and the model config embedded:

```bash
python3 -m venv venv && . venv/bin/activate
pip install torch safetensors huggingface_hub pyyaml   # what convert_ckpt.py needs
python scripts/convert_ckpt.py --ckpt NX-AI/TiRex-2 --out tirex2.safetensors
python scripts/convert_ckpt.py --ckpt NX-AI/TiRex-2-gifteval-zs --out TiRex-2-gifteval-zs.safetensors
```

## 4. Running jitrex

Every entry point runs on the TornadoVM JVM with the SDK's argfile:

```bash
JITREX="java @$TORNADOVM_HOME/tornado-argfile --add-modules jdk.incubator.vector -cp target/classes"
```

### From Java

```java
try (TiRex2Engine engine = TornadoTiRex2Engine.load(Path.of("tirex2.safetensors"))) {
    List<TimeSeries> series = List.of(
            TimeSeries.univariate(values),                        // float[] history, NaN = missing
            new TimeSeries(targets, pastCovariates, futureCovariates));  // [V][T], [Vp][T], [Vf][T+H]
    List<Forecast> forecasts = engine.forecast(series, 64);       // horizon 64
    float[] median = forecasts.get(0).median(0);                  // values[target][quantile][step]
}
```

`CpuTiRex2Engine` is the same API on the CPU reference. Engine options are JVM properties (`-Dtirex.tornado.*`, section 6), for example:

```bash
$JITREX -Dtirex.tornado.device=0:0 -Dtirex.tornado.gemm=cublas-auto -Dtirex.tornado.slstmR=fp16 my.App
```

### Benchmark, cold start and harness server

```bash
# the shared benchmark inputs (the same file drives the NXAI baseline)
python scripts/bench_nxai.py workloads --out workloads.safetensors

# latency and throughput: univariate batches and multivariate series × covariates
$JITREX -Dtirex.tornado.gemm=cublas-auto -Dtirex.tornado.slstmR=fp16 ai.beestream.jitrex.bench.TiRexBench \
    --weights tirex2.safetensors --workloads workloads.safetensors --engine tornado \
    --scenarios throughput,multivariate --batches 1,8,64,512 --mv-batches 1,32 --covariates 15 --out results.jsonl

# cold start: process start → first forecast, split into runtime init, load, first and second forecast
BENCH_EXEC_START_NS=$(date +%s%N) $JITREX ai.beestream.jitrex.bench.TiRexColdStart \
    --weights tirex2.safetensors --workloads workloads.safetensors --out cold.jsonl

# forecasts over a local socket, for NXAI's Python harnesses (scripts/tirex_tornado.py is the client)
$JITREX ai.beestream.jitrex.bench.TiRexServer --weights tirex2.safetensors --port-file port.txt
```

`TiRexBench` options: `--engine tornado|cpu`, `--scenarios`, `--batches`, `--mv-batches`, `--covariates`, `--horizon` (64), `--reps` (30), `--warmup` (5), `--budget` (3 s per point), `--max-batch` (1024), `--config` (a label).

### Tests

```bash
# CPU reference and host pipeline against the goldens (skipped unless both paths are set)
./mvnw test -Dtirex.weights=tirex2.safetensors -Dtirex.golden=golden-cpu
# device: each kernel against the CPU operator, and the whole engine against the goldens
./mvnw test -Paccel-tests -Dtest='*AccelTest' -Dtirex.tornado.device=1:0 \
    -Dtirex.weights=tirex2.safetensors -Dtirex.golden=golden-cpu
```

The goldens come from `scripts/dump_golden.py`, which needs NXAI's package: `git clone https://github.com/NX-AI/tirex-2 && pip install -e ./tirex-2`.

## 5. How the implementation was validated

Validation went in layers, each checked against the one before it:

1. **Executable spec.** `scripts/spec_forward.py` rewrites NXAI's `forward()` using only basic PyTorch ops, without the xlstm or FlashRNN kernels. It is checked stage by stage against NXAI's own activations. It pinned down details the code doesn't state outright: the sLSTM gate order, the mLSTM stabiliser's initial-state term, qk-norm's epsilon, and so on.
2. **Goldens.** `scripts/dump_golden.py` runs NXAI's model (FP32, on the CPU) on 13 cases. It saves the inputs, the batch passed to `forward()`, every stage's activations, the raw output and the final forecasts. The cases cover NXAI's three reference scenarios (univariate, multivariate, multi-target) plus sign-flip TTA, short, full and overlong contexts, trend differencing, binary series, NaN gaps, ragged batches, and past and future covariates in mixed batches. A second golden set comes from the GIFT-Eval checkpoint (285 patches, no qk-norm, sign flip on).
3. **Java CPU reference** (`cpu.CpuReference`). The assembled batch is bit-identical; every forward stage is within 5.5e-6 relative error, and the final forecasts within 2.3e-6.
4. **Kernel parity** (`KernelParityAccelTest`). Each device kernel runs on random inputs against the CPU operator it replaces, with 1e-4 relative tolerance. The inputs are ragged against the tile sizes, and every flag combination and FP16 twin output is covered.
5. **Device engine against the goldens** (`TornadoForwardAccelTest`, and `PlanReuseAccelTest` for plans reused with new inputs). All 13 cases pass, with these worst relative errors:

| Configuration | Model output | Forecasts |
|---|---|---|
| Metal / OpenCL, FP32 (`tiled`) | 2.2e-5 | 8e-6 |
| CUDA, `cublas` (FP32) | 2.7e-6 | 1.4e-6 |
| CUDA, `cublas-auto` + `slstmR=fp16` (best) | 1.5e-4 | 4.7e-5 |
| FP16 on every GEMM (`cublas-fp16`, `half`, `cutile`) | 1.2–1.4e-3 | 5.6–6.7e-4 |
| GIFT-Eval checkpoint, Metal FP32 | 2.2e-5 | 8.2e-6 |
| *NXAI's own CUDA path, against its CPU reference* | *7.7e-3* | |

6. **Memory safety.** `compute-sanitizer --tool memcheck` is clean on CUDA.
7. **Benchmark accuracy parity.** NXAI's own harnesses were run twice, unchanged apart from one patch that routes `TiRex2.predict` to jitrex over a local socket (`scripts/tirex_tornado.py`, `run_with_tornado.py`, and `gift_resumable.py` for GIFT-Eval). Data loading, windowing and metrics are NXAI's, so the scores differ only by the engine.

| Benchmark | Configurations | jitrex / NXAI (geometric mean over tasks) |
|---|---|---|
| fev-bench (`tasks-mini`, 20 tasks) | FP32 and best | SQL 1.00029, MASE 1.00035, WQL 1.00022–1.00039, WAPE 0.9998–1.00006 |
| GIFT-Eval zero-shot, univariate (97 configurations) | FP32 | MASE 0.99980, CRPS 0.99941, sMAPE 0.99987 (96/97; `bizitobs_service/10S/short` ran out of device memory) |

Per task, the two engines differ by up to 1.5%. That gap is NXAI's bf16 CUDA path, not jitrex. On the five fev-bench and three GIFT-Eval tasks with the largest gaps, NXAI run in FP32 on the CPU matches jitrex FP32 to ≤ 4.7e-6 relative, while NXAI's CUDA scores sit 0.1–1.5% away from its own CPU scores.

## 6. Engine options

### JVM properties (`-D…`)

| Property | Values (default first) | Effect |
|---|---|---|
| `tirex.tornado.device` | `<backend>:<device>` (TornadoVM's default device) | Device to run on, numbered as in `tornado --devices` |
| `tirex.tornado.gemm` | `tiled`, `half`, `cublas`, `cublas-tf32`, `cublas-fp16`, `cublas-auto`, `cutile` | GEMM implementation (see *GEMM modes*) |
| `tirex.tornado.fp16MinTokens` | `1024` | `cublas-auto`: GEMMs over at least this many rows use FP16, smaller ones FP32 |
| `tirex.tornado.fp16Kernel` | `gemmex-mixed`, `gemmex-h16`, `lt-h16` | cuBLAS call for an FP16 GEMM: GemmEx FP16→FP32, GemmEx all-FP16, or cuBLASLt all-FP16 (the last two with an FP32 epilogue kernel) |
| `tirex.tornado.slstmR` | `fp32`, `fp16` | Precision of the sLSTM recurrent weights. FP16 halves the bytes each step streams, and the state stays FP32 |
| `tirex.tornado.slstmRows` | `0` (auto) | Rows per sLSTM work-group. Auto picks the largest value that still gives 64 work-groups |
| `tirex.tornado.fuse` | `true`, `false` | Fused epilogues: bias + SiLU after the GEMM, residual + RMSNorm, sLSTM head norm + residual + RMSNorm |
| `tirex.tornado.cudaGraph` | `true` for `cublas*` / `cutile`, `false` otherwise | Capture and replay each task graph as a CUDA graph |

TornadoVM properties used alongside: `tornado.device.memory` (device heap, e.g. `15GB`), `tornado.cuda.nvcc` (for `cutile`).

### GEMM modes

| Mode | Kernel | Precision | Backends |
|---|---|---|---|
| `tiled` | `TiRexKernels.gemm` (KernelContext) | FP32 | all |
| `half` | `TiRexKernels.gemmHalf` (KernelContext, register-blocked) | FP16 in, FP32 accumulation and output | all |
| `cublas` / `cublas-tf32` | cuBLAS SGEMM | FP32 / TF32 | CUDA |
| `cublas-fp16` | cuBLAS GemmEx on every GEMM | FP16 in, FP32 accumulation | CUDA |
| `cublas-auto` | FP16 GemmEx from `fp16MinTokens` rows, SGEMM below | mixed | CUDA |
| `cutile` | `TiRexKernels.gemmHalfTile` (CUDA Tile, tensor cores) | FP16 in, FP32 accumulation | CUDA with toolkit 13.3+ |

### Programmatic options
- `ForecastPipeline(config, forward, batchSize, ttaDiff, ttaSignFlip)`: series per forward (512); trend differencing and sign flip, defaulting to the checkpoint's values.
- `TornadoForward(config, weights, maxCachedPlans)`: the number of cached plans, one per batch shape (8).

### Recommendations

| Platform | Recommended flags | Why |
|---|---|---|
| **NVIDIA** (tested: RTX 5070 Ti) | `-Dtirex.tornado.gemm=cublas-auto -Dtirex.tornado.slstmR=fp16` (CUDA graphs on by default), plus the cuBLAS 13 shim on machines that also have CUDA 12 cuBLAS | Fastest at every batch size: 2.41 ms at batch 1, ~4,850 series/s at batch 512. Error 1.5e-4 / 4.7e-5 |
| NVIDIA, closest to FP32 | `-Dtirex.tornado.gemm=cublas` | FP32 GEMMs, error 2.7e-6 / 1.4e-6; slower than `cublas-auto` at large batches, where the FP16 tensor-core GEMMs pay off |
| NVIDIA, no vendor libraries | `-Dtirex.tornado.gemm=cutile` (toolkit 13.3+), otherwise `half` | `cutile`: 3.1 ms at batch 1, ~1,630 series/s at batch 256. `half`: 7.5 ms, ~1,100 series/s |
| **Apple Silicon** (tested: M4 Pro), Metal or OpenCL | `-Dtirex.tornado.gemm=half -Dtirex.tornado.slstmR=fp16`, on either backend | `half` is 1.9–2.5× faster than `tiled` from batch 8 up; Metal and OpenCL are within ~5% of each other; see the table below |
| Other OpenCL devices | `-Dtirex.tornado.gemm=half` (or `tiled` for FP32) | Portable; not tuned |
| Long contexts (GIFT-Eval checkpoints, 285 patches) | the platform's flags, plus batches of ≤ 32 series and `-Dtirex.tornado.cudaGraph=false` | Device memory grows with rows × patches, and CUDA graphs can fail after many plan rebuilds (a TornadoVM issue under investigation) |

Apple M4 Pro, TornadoVM 7.0.1, JDK 25, end-to-end `forecast()` in ms (the range over two interleaved runs; lower is better):

| Backend | `gemm` | `slstmR` | univariate bs 1 | bs 8 | bs 64 | multivariate 1 × 15 |
|---|---|---|---|---|---|---|
| Metal | `tiled` | `fp32` | 31.2–49.9 | 108.3–111.0 | 738.4–759.4 | 439.9–449.6 |
| Metal | `tiled` | `fp16` | 31.2–33.9 | 106.2–110.6 | 713.8–757.9 | 435.3–446.7 |
| Metal | `half` | `fp32` | 30.3–36.8 | 59.0–59.2 | 301.6–330.9 | 211.7–213.5 |
| Metal | `half` | `fp16` | **27.8–33.0** | **56.9–58.0** | **300.4–302.8** | **199.2–202.2** |
| OpenCL | `tiled` | `fp32` | 33.3–33.4 | 108.4–113.4 | 718.5–731.4 | 426.3–440.7 |
| OpenCL | `tiled` | `fp16` | 32.6–33.9 | 110.4–116.0 | 735.8–775.1 | 422.4–443.0 |
| OpenCL | `half` | `fp32` | 30.2–30.7 | 62.2–62.3 | 286.8–305.6 | 203.4–215.1 |
| OpenCL | `half` | `fp16` | **28.0–28.9** | **58.3–59.1** | **293.1–304.5** | **200.6–207.5** |

The portable kernels are not tuned for Apple GPUs yet: the GEMMs have no vectorised FP16 loads and no simdgroup matrix instructions. NXAI's engine was not benchmarked on MPS.

## 7. Performance comparison with NXAI's engine

**Setup.** An RTX 5070 Ti (sm_120), driver 580.142, JDK 25.0.2, a TornadoVM 7.0.2 development build with the CUDA backend, and jitrex with `cublas-auto` + `slstmR=fp16` and CUDA graphs. NXAI is tirex-2 0.2.1 on torch 2.14.0+cu130, each point taking the best of eager, `torch.compile` and FlexAttention (multivariate only).

**Method**
- **Same inputs:** both engines read the same workload file (`bench_nxai.py workloads`) with the same scenarios and horizon 64. The univariate scenario takes batches of 1 to 1,024 series of 2,048 steps. The multivariate scenario takes 1 or 32 series with 1–63 covariates, half of them future-known.
- **Timing:** `e2e` is the whole `forecast()` call, including host pre- and post-processing. `fwd` is the forward pass alone. Each point is the median over 2 rounds of each run's median (≥ 30 calls or 3 s), after warm-up. The engines alternate per round, and GPU jobs run one at a time.
- **Resources:** each run records its wall time, CPU time, peak host RSS and peak GPU memory above idle.
- **Cold start:** process start to the first forecast (one series, horizon 64), with warm and cold compilation caches (TornadoVM's code cache; torch's inductor and FlashRNN caches).
- **Scripts:** `scripts/bench_nxai.py` (NXAI), `bench.TiRexBench` and `bench.TiRexColdStart` (jitrex), `scripts/cyclone/matrix.sh` (runs both) and `summarize_matrix.py`. The full tables are in `results/2026-09-29-cyclone/summary.md`.

**Results** (end to end, NXAI's best configuration per point)

| Univariate batch | NXAI | jitrex | Speed-up |
|---|---|---|---|
| 1 | 9.70 ms | **2.41 ms** | 4.0× |
| 8 | 15.3 ms | **5.27 ms** | 2.9× |
| 16 | 19.4 ms | **4.74 ms** | 4.1× |
| 64 | 48.2 ms | **14.1 ms** | 3.4× |
| 512 | 361 ms | **105 ms** | 3.4× (4,858 vs 1,417 series/s) |
| 1,024 | 723 ms | **212 ms** | 3.4× |

| Multivariate (series × covariates) | NXAI | jitrex | Speed-up |
|---|---|---|---|
| 1 × 1 | 21.2 ms | **6.42 ms** | 3.3× |
| 1 × 15 | 30.8 ms | **10.2 ms** | 3.0× |
| 1 × 63 | 62.8 ms | **30.2 ms** | 2.1× |
| 32 × 15 | 487 ms | **205 ms** | 2.4× |
| 32 × 63 | 1,869 ms | **920 ms** | 2.0× |

| Cold start (median) | NXAI | jitrex |
|---|---|---|
| Warm caches: process start → first forecast | 8.18 s | **1.41 s** |
| Cold caches | 20.8 s | **1.80 s** |
| Second forecast | 12.3 ms | **4.2 ms** |

Peak GPU memory is comparable (jitrex 5–9.5 GB, NXAI 9.5–9.7 GB). jitrex uses more host memory (3–9.6 GB RSS against 2 GB), because it keeps host copies of the weights.

On the benchmark suites (section 5), fev-bench mini took 82.7 s of inference with jitrex against 185.9 s with NXAI.

## 8. Kernels

All kernels are in [`TiRexKernels.java`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java), written with KernelContext, except `gemmHalfTile`, which uses TileContext. The *Hybrid alternative* column says what can replace a kernel on CUDA.

| Kernel | What it computes | Hybrid alternative |
|---|---|---|
| [`scalePatch`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L46) | Per-row NaN-aware scaler (nanmean/std, arcsinh, binary rows), patching and the is-present mask; FP16 twin | — |
| [`gemm`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L129) | Tiled FP32 GEMM `C (+)= act(A·Wᵀ + b)` with bias / SiLU / accumulate epilogue | cuBLAS `cublasSgemm`, `cublasSgemmTF32` |
| [`gemmHalf`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L186) | Register-blocked FP16-input GEMM, FP32 accumulation, same epilogues and output twin | cuBLAS `cublasGemmExFP16FP32`, `cublasGemmExFP16`; cuBLASLt `ltMatmulFP16`; CUDA Tile `gemmHalfTile` |
| [`gemmHalfTile`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L301) | The FP16 GEMM in the CUDA Tile API (`mma` on tensor cores, masked tiles) | is itself the CUDA-only variant |
| [`rmsNorm`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L327) | RMSNorm per row; FP16 twin | — |
| [`headLayerNorm`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L360) | Per-head LayerNorm: plain, gated by σ(o) (mLSTM output), or added into the residual stream | — |
| [`mlstmCell`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L415) | mLSTM cell in the stabilised parallel form, with suffix-sum log-decays (up to 512 patches) | — |
| [`slstmRecurrence`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L497) | sLSTM recurrence over all steps, one work-group per (row block, head), each R read shared by the rows | — |
| [`slstmRecurrenceHalfR`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L574) | The same with FP16 recurrent weights | — |
| [`swiGlu`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L649) | `SiLU(gate) · up` for the fused up-gate projection; FP16 twin | — |
| [`unpatch`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L668) | Patches back to a sequence, then `sinh(clamp(y))·scale + loc` | — |
| [`gatherRows`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L693) | Gathers the known-covariate rows, time-reversed for the reverse cell pass; FP16 twin | — |
| [`scatterRows`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L710) | Scatters the combined rows back | — |
| [`concatForward`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L725) | Forward and reverse outputs, concatenated for the combination layer; FP16 twin | — |
| [`addInPlace`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L746) | Residual add | — |
| [`qkNorm`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L762) | Weightless RMS normalisation of q and k per head, several vectors per pass | — |
| [`variateAttention`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L805) | Flash-style attention across the rows of each group, masked (covariates don't attend to targets), costing O(Σ group²); FP16 twin | — |
| [`biasFill`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L895) | Writes the bias into C before a cuBLAS GEMM with β = 1 | helper for the cuBLAS path |
| [`siluInPlace`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L905) | SiLU after a cuBLAS GEMM; FP16 twin | helper for the cuBLAS path |
| [`toHalf`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L917) | FP32 → FP16 for a GEMM input without a twin | helper for the FP16 GEMMs |
| [`fromHalfEpilogue`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L927) | FP16 C → FP32 with bias / SiLU / accumulate, after an all-FP16 cuBLAS GEMM | helper for `gemmex-h16`, `lt-h16` |
| [`biasSilu`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L953) | Fused bias + SiLU + FP16 twin after a cuBLAS GEMM | helper for the cuBLAS path (`fuse`) |
| [`addRmsNorm`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L978) | Fused residual add + RMSNorm; FP16 twin | — (`fuse`) |
| [`headNormAddRmsNorm`](src/main/java/ai/beestream/jitrex/tornado/TiRexKernels.java#L1018) | Fused sLSTM head LayerNorm + residual add + RMSNorm; FP16 twin (needs 512 threads per work-group) | — (`fuse`) |

An *FP16 twin* is an FP16 copy of a kernel's output that the kernel writes for the next GEMM. With twins, the FP16 GEMM modes need no separate conversion pass.

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
| `tornado` | `TiRexKernels` (every device kernel, section 8), `PackedWeights` (one device buffer plus per-matrix FP32/FP16 copies for the library GEMMs; fused q\|k\|v\|o\|i\|f, up_gate\|up and the block-diagonal sLSTM gates), `TornadoForward` (a plan per batch shape — rows, known covariates, kept rows — built as a chain of task graphs: input, one per block, output; the GEMM modes; CUDA graphs), `TornadoTiRex2Engine` |

## Scripts (`scripts/`)

| Script | What it does |
|---|---|
| `convert_ckpt.py` | `model.ckpt` + `model-config.yaml` → one `tirex2.safetensors` (FP32, aliases removed, config in the metadata) |
| `spec_forward.py` | An explicit PyTorch spec of `forward()` (basic ops only, no xlstm/flashrnn) checked stage by stage against the goldens; the Java reference is a port of it |
| `dump_golden.py` | Runs the NXAI model on 13 cases and saves the inputs, the batch passed to `forward()`, the activations of every stage, the raw model output and the final forecasts |
| `bench_nxai.py` | NXAI baseline: `workloads` (shared inputs), `run` (latency/throughput, e2e + forward-only), `coldstart` |
| `cyclone_p0.sh` | The whole reference setup on a CUDA box: venv, CUDA-kernel check, conversion, CPU + CUDA goldens, NXAI baseline, cold start |
| `tirex_tornado.py`, `run_with_tornado.py` | Run an unmodified NXAI harness with `TiRex2.predict` served by `TiRexServer` |
| `gift_resumable.py` | GIFT-Eval one configuration at a time: results appended as they land, resumable, server restart and retry on failure |
| `cyclone/` | The benchmark and experiment runners used on the RTX 5070 Ti: `matrix.sh` + `summarize_matrix.py` (section 7), `acc_setup.sh` / `acc_run.sh` / `acc_compare.py` (section 5), `profile.sh` + `kern_sum.py` (nsys), A/B scripts |

```bash
./scripts/cyclone_p0.sh ~/tirex2
```
