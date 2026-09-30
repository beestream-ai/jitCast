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
- **Fast.** On an RTX 5070 Ti it is 2.0–4.1× faster than NXAI's PyTorch engine end to end, and it
  starts 5.8× faster.

## Where it runs

| Platform | Backend | Status | Recommended flags |
|---|---|---|---|
| NVIDIA GPU | CUDA | validated, benchmarked (RTX 5070 Ti) | `-Dtirex.tornado.gemm=cublas-auto -Dtirex.tornado.slstmR=fp16` |
| Apple Silicon | Metal or OpenCL | validated (M4 Pro) | `-Dtirex.tornado.gemm=half -Dtirex.tornado.slstmR=fp16` |
| Other OpenCL GPUs | OpenCL | expected to work, not tested | `-Dtirex.tornado.gemm=half` |
| Any CPU | plain JVM | FP32 reference, for validation | — |

Every option, the GEMM modes and the hybrid APIs are in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

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
     -Dtirex.tornado.device=0:0 -Dtirex.tornado.gemm=cublas-auto -Dtirex.tornado.slstmR=fp16 your.Main
```

Covariates, the benchmark and cold-start tools, the harness server and the tests are covered in
[docs/USAGE.md](docs/USAGE.md).

## Performance

End-to-end `forecast()` on an RTX 5070 Ti, against the best of NXAI's eager, `torch.compile` and
FlexAttention configurations:

| Workload | NXAI | jitrex | Speed-up |
|---|---|---|---|
| 1 series | 9.70 ms | **2.41 ms** | 4.0× |
| 64 series | 48.2 ms | **14.1 ms** | 3.4× |
| 512 series | 361 ms | **105 ms** | 3.4× (4,858 vs 1,417 series/s) |
| 1 series × 15 covariates | 30.8 ms | **10.2 ms** | 3.0× |
| 32 series × 63 covariates | 1,869 ms | **920 ms** | 2.0× |
| 32 series, GIFT-Eval checkpoint (285 patches) | 130.8 ms | **73.5 ms** | 1.8× |
| Cold start: process → first forecast | 8.18 s | **1.41 s** | 5.8× |

The method, the full tables, memory use and the Apple Silicon numbers are in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Documentation

| Document | Contents |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | TornadoVM SDKs, the NVIDIA extras (cuBLAS 13, CUDA Tile), building, converting weights |
| [USAGE.md](docs/USAGE.md) | The Java API, benchmark, cold start, harness server, tests |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Every engine option, GEMM modes, hybrid APIs, per-platform recommendations |
| [VALIDATION.md](docs/VALIDATION.md) | How the implementation was checked, from an executable spec to GIFT-Eval and fev-bench |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | The comparison with NXAI's engine: method and results |
| [KERNELS.md](docs/KERNELS.md) | All 25 kernels, with source links and their hybrid alternatives |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Architecture, code layout and scripts |

## License

jitrex is released under the [Apache License, Version 2.0](LICENSE).
Copyright (c) 2026 Beestream Ltd.
`util/Json.java` is derived from [jitllm](https://github.com/beehive-lab/jitllm) (MIT); see [NOTICE](NOTICE).
The TiRex-2 model and its weights are NX-AI's, under their own licence terms.
