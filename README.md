<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/jitcast-logo-dark.svg">
    <img alt="jitcast" src="docs/images/jitcast-logo-light.svg" width="480">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Java 22+" src="https://img.shields.io/badge/java-22%2B-orange.svg">
  <img alt="TornadoVM" src="https://img.shields.io/badge/TornadoVM-7.1.0%2B-0B1F33.svg">
</p>

# jitcast

**jitcast** is an inference engine for zero-shot time-series forecasting models, written in Java on
[TornadoVM](https://github.com/beehive-lab/TornadoVM) (similar to [jitLLM](https://github.com/beehive-lab/jitllm)).
It runs [TiRex-2](https://github.com/NX-AI/tirex-2), NXAI's model,
[Timestral](https://huggingface.co/mistralai/Timestral-5M), Mistral AI's family (see
[docs/TIMESTRAL.md](docs/TIMESTRAL.md)), and [Chronos-2](https://huggingface.co/amazon/chronos-2),
Amazon's model (see [docs/CHRONOS2.md](docs/CHRONOS2.md)).

For TiRex-2, jitcast reproduces NXAI's `ForecastModel.forecast()` end to end: 9-quantile forecasts for
univariate series, multi-target series, and series with past or future covariates. The host
pipeline is in Java, and the model's forward pass runs on the GPU through TornadoVM.

- **Portable.** Every kernel is a TornadoVM KernelContext kernel, so the same code runs on CUDA,
  OpenCL and Metal.
- **Hybrid on NVIDIA.** The GEMMs can run on cuBLAS / cuBLASLt or on a CUDA Tile kernel, and each
  task graph can be replayed as a CUDA graph.
- **Validated against NXAI.** On fev-bench (100 tasks) and GIFT-Eval (97 configurations), jitcast's
  aggregate scores are within 0.03% and 0.08% of NXAI's on the same GPU.

## Where it runs

| Platform | Backend | Status | Recommended flags |
|---|---|---|---|
| NVIDIA GPU | CUDA | validated, benchmarked (A100) | `-Dtirex.tornado.gemm=cublas-auto` |
| NVIDIA GPU | OpenCL | validated (A100); slower than CUDA | `-Dtirex.tornado.gemm=half` |
| Apple Silicon | Metal or OpenCL | validated (M4 Pro) | `-Dtirex.tornado.gemm=half` |
| Other OpenCL GPUs (Intel, AMD) | OpenCL | expected to work, not tested | `-Dtirex.tornado.gemm=half` |
| Any CPU | plain JVM | FP32 reference, for validation | — |

Every option, the GEMM modes and the hybrid APIs are in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Quick start

You need JDK 22+, a TornadoVM SDK (7.1.0 or newer, `jdk22plus` family) and, once, Python 3.11+
with PyTorch to convert the weights. The full guide is in [docs/INSTALL.md](docs/INSTALL.md).

```bash
# 1. a JDK and TornadoVM (or download an SDK from tornadovm.org/downloads)
sdk install java 25.0.4-tem                     # any JDK 22+; the TornadoVM SDK does not include one
sdk install tornadovm 7.1.0-jdk22plus-cuda      # or ...-metal / ...-opencl
tornado --devices                               # device ids, as <backend>:<device>

# 2. jitcast
git clone https://github.com/beestream-ai/jitcast && cd jitcast
./mvnw -q package -DskipTests

# 3. weights: NXAI's checkpoint from Hugging Face, converted to safetensors
python3.11 -m venv venv && . venv/bin/activate   # Python 3.11 or newer
pip install torch safetensors numpy huggingface_hub pyyaml
python scripts/convert_ckpt.py --ckpt NX-AI/TiRex-2 --out tirex2.safetensors
```

Forecasting from Java:

```java
try (ForecastEngine engine = TornadoTiRex2Engine.load(Path.of("tirex2.safetensors"))) {
    List<Forecast> forecasts = engine.forecast(List.of(TimeSeries.univariate(history)), 64);
    float[] median = forecasts.get(0).median(0);        // values[target][quantile][step]
}
```

Run it on the TornadoVM JVM, choosing the device and flags:

```bash
java @$TORNADOVM_HOME/tornado-argfile --add-modules jdk.incubator.vector -cp target/classes:<your app> \
     -Djitcast.tornado.device=0:0 -Dtirex.tornado.gemm=cublas-auto your.Main
```

Covariates, the benchmark and cold-start tools, the harness server and the tests are covered in
[docs/USAGE.md](docs/USAGE.md).

## Performance

End-to-end `forecast()` on an NVIDIA A100 80GB, jitcast's best configuration (`cublas-auto`) against
NXAI's best of eager, `torch.compile` and FlexAttention on the same GPU, on TornadoVM 7.1.0 (2026-10-05).

| Workload | NXAI | jitcast | Speed-up |
|---|---|---|---|
| 1 series | 35.1 ms | **2.25 ms** | 15.6× |
| 64 series | 123.7 ms | **14.8 ms** | 8.3× |
| 512 series | 620 series/s | **4,863 series/s** | 7.8× |
| 1 series × 15 covariates | 72.1 ms | **8.88 ms** | 8.1× |
| 32 series × 63 covariates | 3,404 ms | **837.3 ms** | 4.1× |
| 1 series, GIFT-Eval checkpoint (285 patches) | 98.8 ms | **16.3 ms** | 6.1× |
| 32 series, GIFT-Eval checkpoint | 191.6 ms | **58.6 ms** | 3.3× |
| Cold start: process → first forecast (warm caches) | 13.85 s | **3.75 s** | 3.7× |
| fev-bench, 100 tasks: inference time | 2,897 s | **919 s** | 3.2× |

![jitcast speed-up over NXAI on the A100](docs/images/eval-a100.svg)

The other configurations (cuBLAS FP32, TornadoVM's own FP16 GEMMs), the long-context, cold-start
and benchmark-suite tables and the method are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Documentation

| Document | Contents |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | TornadoVM SDKs, the NVIDIA extras (cuBLAS 13, CUDA Tile), building, converting weights |
| [USAGE.md](docs/USAGE.md) | The Java API, benchmark, cold start, harness server, tests |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Every engine option, GEMM modes, hybrid APIs, per-platform recommendations |
| [VALIDATION.md](docs/VALIDATION.md) | How the implementation was checked, from an executable spec to GIFT-Eval and fev-bench |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | The comparison with NXAI's engine on an NVIDIA A100: method, results, charts |
| [TIMESTRAL.md](docs/TIMESTRAL.md) | Timestral: quick start, how it maps onto jitcast, validation, performance on Apple silicon, options |
| [CHRONOS2.md](docs/CHRONOS2.md) | Chronos-2: quick start, how it maps onto jitcast, validation, performance on Apple silicon, options |
| [KERNELS.md](docs/KERNELS.md) | Every kernel, with source links and their hybrid alternatives |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Architecture, code layout and scripts |
| [ROADMAP.md](docs/ROADMAP.md) | Planned work, and the TornadoVM issues that limit jitcast today |

## License

jitcast is released under the [Apache License, Version 2.0](LICENSE).
Copyright (c) 2026 Beestream Ltd.
`util/Json.java` is derived from [jitllm](https://github.com/beehive-lab/jitllm) (MIT); see [NOTICE](NOTICE).
The TiRex-2 model and its weights are NX-AI's, under their own licence terms. The Timestral weights
are Mistral AI's, under a Modified MIT licence with a revenue restriction (see
[docs/TIMESTRAL.md](docs/TIMESTRAL.md#licence)). The Chronos-2 weights are Amazon's, under the
Apache License 2.0.
