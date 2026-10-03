<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/jitrex-logo-dark.svg">
    <img alt="jitrex" src="docs/images/jitrex-logo-light.svg" width="480">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Java 22+" src="https://img.shields.io/badge/java-22%2B-orange.svg">
  <img alt="TornadoVM" src="https://img.shields.io/badge/TornadoVM-7.0.1%2B-0B1F33.svg">
</p>

# jitrex

**jitrex** is an inference engine for [TiRex-2](https://github.com/NX-AI/tirex-2), NXAI's zero-shot
time-series forecasting model, written in Java on [TornadoVM](https://github.com/beehive-lab/TornadoVM) (similar to [jitLLM](https://github.com/beehive-lab/jitllm)).

jitrex reproduces NXAI's `ForecastModel.forecast()` end to end: 9-quantile forecasts for
univariate series, multi-target series, and series with past or future covariates. The host
pipeline is in Java, and the model's forward pass runs on the GPU through TornadoVM.

- **Portable.** Every kernel is a TornadoVM KernelContext kernel, so the same code runs on CUDA,
  OpenCL and Metal.
- **Hybrid on NVIDIA.** The GEMMs can run on cuBLAS / cuBLASLt or on a CUDA Tile kernel, and each
  task graph can be replayed as a CUDA graph.
- **Validated against NXAI.** On fev-bench and GIFT-Eval, jitrex gives the same scores as NXAI's
  FP32 reference to about 1e-6.
- **Fast.** On an RTX 5070 Ti it is 2.0–6.8× faster than NXAI's PyTorch engine end to end, and it
  starts 4.0× faster; on an A100 it is 3.3–13.1× faster and starts 2.8× faster. On an Apple M4 Pro
  it is up to 3.3× faster than NXAI on MPS (6.4× at long context), and on par at the largest
  multivariate batches.

## Where it runs

| Platform | Backend | Status | Recommended flags |
|---|---|---|---|
| NVIDIA GPU | CUDA | validated, benchmarked (RTX 5070 Ti, A100) | `-Dtirex.tornado.gemm=cublas-auto` |
| NVIDIA GPU | OpenCL | validated, benchmarked (RTX 5070 Ti, A100); slower than CUDA | `-Dtirex.tornado.gemm=half` |
| Apple Silicon | Metal or OpenCL | validated, benchmarked (M4 Pro) | `-Dtirex.tornado.gemm=half` |
| Other OpenCL GPUs (Intel, AMD) | OpenCL | expected to work, not tested | `-Dtirex.tornado.gemm=half` |
| Any CPU | plain JVM | FP32 reference, for validation | — |

On an A100 (and other sm_80 GPUs) also set `-Dtirex.tornado.slstmR=fp16`: the default sLSTM kernel needs
more registers per work-group than the GPU has. Every option, the GEMM modes and the hybrid APIs are in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Quick start

You need JDK 22+, a TornadoVM SDK (7.0.1 or newer, `jdk22plus` family) and, once, Python with
PyTorch to convert the weights. The full guide is in [docs/INSTALL.md](docs/INSTALL.md).

```bash
# 1. TornadoVM (or download an SDK from tornadovm.org/downloads)
sdk install tornadovm 7.0.1-jdk22plus-cuda      # or ...-metal / ...-opencl
tornado --devices                               # device ids, as <backend>:<device>

# 2. jitrex
git clone https://github.com/beestream-ai/jitcast && cd jitrex
./mvnw -q package -DskipTests

# 3. weights: NXAI's checkpoint from Hugging Face, converted to safetensors
pip install torch safetensors huggingface_hub pyyaml
python scripts/convert_ckpt.py --ckpt NX-AI/TiRex-2 --out tirex2.safetensors
```

Forecasting from Java:

```java
try (TiRex2Engine engine = TornadoTiRex2Engine.load(Path.of("tirex2.safetensors"))) {
    List<Forecast> forecasts = engine.forecast(List.of(TimeSeries.univariate(history)), 64);
    float[] median = forecasts.get(0).median(0);        // values[target][quantile][step]
}
```

Run it on the TornadoVM JVM, choosing the device and flags:

```bash
java @$TORNADOVM_HOME/tornado-argfile --add-modules jdk.incubator.vector -cp target/classes:<your app> \
     -Dtirex.tornado.device=0:0 -Dtirex.tornado.gemm=cublas-auto your.Main
```

Covariates, the benchmark and cold-start tools, the harness server and the tests are covered in
[docs/USAGE.md](docs/USAGE.md).

## Performance

End-to-end `forecast()`, jitrex's best configuration against NXAI's best on the same GPU
(NVIDIA: `cublas-auto` against the best of eager, `torch.compile` and FlexAttention; Apple: Metal `half` against eager MPS).
The RTX 5070 Ti ran a TornadoVM 7.0.2 development build (2026-10-01); the A100 and the M4 Pro ran TornadoVM
develop (2026-10-03).

| Workload | RTX 5070 Ti: NXAI | jitrex | Speed-up | A100: NXAI | jitrex | Speed-up | M4 Pro: NXAI | jitrex | Speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 series | 9.63 ms | **1.42 ms** | 6.8× | 36.5 ms | **2.78 ms** | 13.1× | 88.1 ms | **27.1 ms** | 3.3× |
| 64 series | 48.6 ms | **13.7 ms** | 3.6× | 127.9 ms | **15.4 ms** | 8.3× | 357.5 ms | **277.4 ms** | 1.3× |
| 512 series | 1,419 series/s | **4,716 series/s** | 3.3× | 606 series/s | **4,320 series/s** | 7.1× | 223 series/s | **248 series/s** | 1.1× |
| 1 series × 15 covariates | 30.7 ms | **7.83 ms** | 3.9× | 75.1 ms | **9.66 ms** | 7.8× | 293.9 ms | **198.0 ms** | 1.5× |
| 32 series × 63 covariates | 1,870 ms | **936.5 ms** | 2.0× | 3,236 ms | **913.2 ms** | 3.5× | 15,968 ms | 17,433 ms | 0.9× |
| 1 series, GIFT-Eval checkpoint (285 patches) | 30.4 ms | **9.05 ms** | 3.4× | 102.5 ms | **18.0 ms** | 5.7× | 522.1 ms | **82.0 ms** | 6.4× |
| 32 series, GIFT-Eval checkpoint | 130.4 ms | **63.3 ms** | 2.1× | 197.1 ms | **60.6 ms** | 3.3× | 1,181 ms | **1,071 ms** | 1.1× |
| Cold start: process → first forecast | 6.65 s | **1.66 s** | 4.0× | 10.39 s | **3.76 s** | 2.8× | 2.04 s | 2.07 s | 1.0× |

![jitrex speed-up over NXAI on the RTX 5070 Ti](docs/images/eval-nvidia.svg)

![jitrex speed-up over NXAI on the A100](docs/images/eval-a100.svg)

The full matrix (CUDA without hybrid, CUDA Tile, cuBLAS, OpenCL on each GPU), the long-context
and cold-start tables and the method are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Documentation

| Document | Contents |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | TornadoVM SDKs, the NVIDIA extras (cuBLAS 13, CUDA Tile), building, converting weights |
| [USAGE.md](docs/USAGE.md) | The Java API, benchmark, cold start, harness server, tests |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Every engine option, GEMM modes, hybrid APIs, per-platform recommendations |
| [VALIDATION.md](docs/VALIDATION.md) | How the implementation was checked, from an executable spec to GIFT-Eval and fev-bench |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | The eight-row comparison with NXAI's engine on NVIDIA and Apple GPUs: method, results, charts |
| [KERNELS.md](docs/KERNELS.md) | All 26 kernels, with source links and their hybrid alternatives |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Architecture, code layout and scripts |

## License

jitrex is released under the [Apache License, Version 2.0](LICENSE).
Copyright (c) 2026 Beestream Ltd.
`util/Json.java` is derived from [jitllm](https://github.com/beehive-lab/jitllm) (MIT); see [NOTICE](NOTICE).
The TiRex-2 model and its weights are NX-AI's, under their own licence terms.
