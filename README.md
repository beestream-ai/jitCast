<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/jitcast-logo-dark.svg">
    <img alt="jitcast" src="docs/images/jitcast-logo-light.svg" width="560">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Java 22+" src="https://img.shields.io/badge/java-22%2B-orange.svg">
  <img alt="TornadoVM" src="https://img.shields.io/badge/TornadoVM-8.0.0%2B-0B1F33.svg">
</p>

# jitcast

**jitcast** runs zero-shot time-series foundation models in Java, on GPUs through
[TornadoVM](https://github.com/beehive-lab/TornadoVM) (a sibling of [jitLLM](https://github.com/beehive-lab/jitllm)).
Each model reproduces its reference engine's `forecast()` end to end — batch assembly, scaling,
decoding, calibration and test-time augmentation on the host, the forward pass on the GPU — behind
one Java API.

| Model | From | Sizes | Forecasts | Inputs | Page |
|---|---|---|---|---|---|
| **TiRex-2** | [NXAI](https://github.com/NX-AI/tirex-2) | 82.5M (xLSTM) | 9 quantiles, up to 320 or 928 steps | univariate, multi-target, past and future covariates | [models/TIREX2.md](docs/models/TIREX2.md) |
| **Timestral** | [Mistral AI](https://huggingface.co/mistralai/Timestral-5M) | 5M, 28M, 123M, 209M (transformer) | 21 quantiles, any horizon | univariate, multivariate, past and known-future covariates | [models/TIMESTRAL.md](docs/models/TIMESTRAL.md) |
| **Chronos-2** | [Amazon](https://huggingface.co/amazon/chronos-2) | 28M, 120M (T5-style encoder) | 13 or 21 quantiles, any horizon | univariate, multivariate, past and known-future covariates, cross-learning | [models/CHRONOS2.md](docs/models/CHRONOS2.md) |

- **Portable.** Every kernel is a TornadoVM KernelContext kernel, so the same code runs on CUDA,
  OpenCL and Metal.
- **Hybrid on NVIDIA.** GEMMs can run on cuBLAS (FP32, TF32 or FP16 tensor cores), and each forward
  pass is replayed as a CUDA graph.
- **Validated against the reference engines.** Every stage of every model is checked against its
  reference's own activations, on CUDA, OpenCL and Metal; TiRex-2's fev-bench and GIFT-Eval scores are within
  0.03% and 0.08% of NXAI's on the same GPU.
- **Faster than the reference engines.** On an A100 and an A10, at every batch size measured, from 1
  to 512 series: 3–16× at one series, where the references' per-call overhead dominates.

## Where it runs

| Platform | Backend | TiRex-2 | Timestral | Chronos-2 |
|---|---|---|---|---|
| NVIDIA GPU | CUDA | validated, benchmarked (A100, A10) | validated, benchmarked (A100, RTX 5070 Ti, A10) | validated, benchmarked (RTX 5070 Ti, A10) |
| NVIDIA GPU | OpenCL | validated (A100) | validated (A100) | not tested |
| Apple Silicon | Metal or OpenCL | validated (M4 Pro) | validated, benchmarked (M4 Pro) | validated, benchmarked (M4 Pro) |
| Other OpenCL GPUs (Intel, AMD) | OpenCL | expected to work, not tested | expected to work, not tested | expected to work, not tested |
| Any CPU | plain JVM | FP32 reference, for validation | FP32 reference, for validation | FP32 reference, for validation |

The options and the recommended flags per platform are in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Quick start

You need JDK 22+ and a TornadoVM SDK (8.0.0 or newer, `jdk22plus` family); the full guide is in
[docs/INSTALL.md](docs/INSTALL.md). TornadoVM 8 compiles kernels with its new TC compiler: up to 1.36×
faster than 7.3.1 at small batches, and it runs the CUDA Tile GEMM.

```bash
# 1. a JDK and TornadoVM (or download an SDK from tornadovm.org/downloads)
sdk install java 25.0.4-tem                     # any JDK 22+; the TornadoVM SDK does not include one
sdk install tornadovm 8.0.0-jdk22plus-cuda      # or ...-metal / ...-opencl
tornado --devices                               # device ids, as <backend>:<device>

# 2. jitcast
git clone https://github.com/beestream-ai/jitcast && cd jitcast
./mvnw -q package -DskipTests

# 3. a model: Timestral and Chronos-2 load as published; TiRex-2 is converted once (see its page)
mkdir -p Timestral-5M && for f in config.json model.safetensors; do
  curl -L -o Timestral-5M/$f https://huggingface.co/mistralai/Timestral-5M/resolve/main/$f
done
```

Forecasting from Java, the same for every model:

```java
try (ForecastEngine engine = TornadoTimestralEngine.load(Path.of("Timestral-5M"))) {  // or TornadoChronos2Engine, TornadoTiRex2Engine
    List<Forecast> forecasts = engine.forecast(List.of(TimeSeries.univariate(history)), 64);
    float[] median = forecasts.get(0).median(0);        // values[target][quantile][step]
}
```

Run it on the TornadoVM JVM, choosing the device:

```bash
java @$TORNADOVM_HOME/tornado-argfile --add-modules jdk.incubator.vector -cp target/classes:<your app> \
     -Djitcast.tornado.device=0:0 your.Main
```

Covariates, the benchmarks, the harness server and the tests are covered in [docs/USAGE.md](docs/USAGE.md).

## Performance

End-to-end `forecast()` on an NVIDIA A100 80GB (TornadoVM `develop`), jitcast's fastest option against
the fastest option of each model's PyTorch reference (eager, `torch.compile`, TF32, BF16 and FP16, and
FlexAttention for TiRex-2), per workload. The method, both times of every point and the other GPUs and
Apple silicon are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md#fastest-against-fastest).

| Model | 1 series | 8 series | 128 series | 512 series | 1 series, H = 1,024 | 8 series, long context |
|---|---|---|---|---|---|---|
| TiRex-2 | **13.1×** | **9.8×** | **7.6×** | **7.7×** | – | **3.6×** |
| Timestral-5M | **10.7×** | **3.7×** | **1.7×** | 0.72× | **9.4×** | **4.4×** |
| Timestral-209M | **6.9×** | **2.9×** | 0.67× | 1.0× | **6.7×** | **1.2×** |
| Chronos-2 small | **5.7×** | **3.0×** | 1.1× | **1.2×** | **4.1×** | 0.98× |
| Chronos-2 | **4.6×** | **2.3×** | 0.86× | 0.96× | **3.1×** | 0.71× |

jitcast wins at small batches, where PyTorch's per-call overhead dominates, and TiRex-2 at every size.
From 128 series up Timestral and Chronos-2 are GEMM-bound on the same tensor cores in both engines.
On fev-bench (100 tasks) TiRex-2's inference time is 3.2× lower than NXAI's.

## Documentation

| Document | Contents |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | TornadoVM SDKs (prebuilt and built from source), device numbering, NVIDIA extras, building, getting the weights |
| [USAGE.md](docs/USAGE.md) | The Java API, the benchmarks, the harness server, the tests |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Every option, GEMM modes, hybrid APIs, per-platform recommendations, per model |
| [models/TIREX2.md](docs/models/TIREX2.md), [models/TIMESTRAL.md](docs/models/TIMESTRAL.md), [models/CHRONOS2.md](docs/models/CHRONOS2.md) | Each model: architecture, checkpoints, features, how it maps onto jitcast, options, licence |
| [VALIDATION.md](docs/VALIDATION.md) | How each model was checked against its reference engine |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | The comparisons with the reference engines: method, results, charts |
| [KERNELS.md](docs/KERNELS.md) | Every kernel, shared and per model, with source links and their hybrid alternatives |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Architecture, code layout, adding a model family, scripts |
| [ROADMAP.md](docs/ROADMAP.md) | Planned work, and the TornadoVM issues that limit jitcast today |

## License

jitcast is released under the [Apache License, Version 2.0](LICENSE).
Copyright (c) 2026 Beestream Ltd.
`util/Json.java` is derived from [jitllm](https://github.com/beehive-lab/jitllm) (MIT); see [NOTICE](NOTICE).
The model weights are their authors': TiRex-2's are NX-AI's, under their own terms; Timestral's are
Mistral AI's, under a Modified MIT licence with a revenue restriction
([models/TIMESTRAL.md](docs/models/TIMESTRAL.md#licence)); Chronos-2's are Amazon's, under the Apache
License 2.0. jitcast does not distribute them.
