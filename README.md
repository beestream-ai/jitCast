<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/jitcast-logo-dark.svg">
    <img alt="jitcast" src="docs/images/jitcast-logo-light.svg" width="560">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Java 22+" src="https://img.shields.io/badge/java-22%2B-orange.svg">
  <img alt="TornadoVM" src="https://img.shields.io/badge/TornadoVM-7.3.1%2B-0B1F33.svg">
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
  reference's own activations, on every backend; TiRex-2's fev-bench and GIFT-Eval scores are within
  0.03% and 0.08% of NXAI's on the same GPU.
- **Faster than the reference engines.** On an A100, at every batch size measured, from 1 to 512
  series: 4–16× at one series, where the references' per-call overhead dominates.

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

You need JDK 22+ and a TornadoVM SDK (7.3.1 or newer, `jdk22plus` family); the full guide is in
[docs/INSTALL.md](docs/INSTALL.md). TornadoVM's `develop` branch, with its new TC compiler, is
recommended: up to 1.36× faster at small batches, and it runs the CUDA Tile GEMM.

```bash
# 1. a JDK and TornadoVM (or download an SDK from tornadovm.org/downloads)
sdk install java 25.0.4-tem                     # any JDK 22+; the TornadoVM SDK does not include one
sdk install tornadovm 7.3.1-jdk22plus-cuda      # or ...-metal / ...-opencl
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

End-to-end `forecast()` against each model's reference engine on the same inputs, on an NVIDIA
A100 80GB. The method and every configuration are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md),
including Apple silicon.

**TiRex-2**, jitcast `cublas-auto` against NXAI's best of eager, `torch.compile` and FlexAttention
(TornadoVM 7.1.0):

| Workload | NXAI | jitcast | Speed-up |
|---|---|---|---|
| 1 series | 35.1 ms | **2.25 ms** | 15.6× |
| 64 series | 123.7 ms | **14.8 ms** | 8.3× |
| 512 series | 620 series/s | **4,863 series/s** | 7.8× |
| 1 series × 15 covariates | 72.1 ms | **8.88 ms** | 8.1× |
| fev-bench, 100 tasks: inference time | 2,897 s | **919 s** | 3.2× |

**Timestral**, jitcast's defaults (cuBLAS FP32, CUDA graphs) against Mistral's reference on PyTorch
CUDA (TornadoVM develop):

| Workload | 5M reference | 5M jitcast | 28M reference | 28M jitcast | 209M reference | 209M jitcast |
|---|---|---|---|---|---|---|
| 1 series | 43.1 ms | **3.8 ms (11.3×)** | 53.1 ms | **6.2 ms (8.6×)** | 88.9 ms | **21.4 ms (4.1×)** |
| 8 series | 45.0 ms | **9.8 ms (4.6×)** | 55.5 ms | **13.9 ms (4.0×)** | 93.0 ms | **41.9 ms (2.2×)** |
| 128 series | 45.5 ms | **26.4 ms (1.7×)** | 94.5 ms | **76.8 ms (1.2×)** | 512.8 ms | **477.6 ms (1.1×)** |
| 512 series | 101.9 ms | **81.5 ms (1.2×)** | 349.1 ms | **284.2 ms (1.2×)** | 2,071.0 ms | **1,802.3 ms (1.1×)** |
| 8 series × (2 past + 2 known) | 50.4 ms | **12.1 ms (4.2×)** | 75.5 ms | **30.2 ms (2.5×)** | 336.7 ms | **162.9 ms (2.1×)** |

With cuBLAS FP16, Timestral-209M is another 3–4× faster at large batches.

**Chronos-2**, jitcast's defaults (cuBLAS FP32, CUDA graphs) against Amazon's reference on PyTorch
CUDA; not measured on the A100 yet (the RTX 5070 Ti and Apple M4 Pro results are in
[PERFORMANCE.md](docs/PERFORMANCE.md#chronos-2)):

| Workload | small reference | small jitcast | base reference | base jitcast |
|---|---|---|---|---|
| 1 series |  |  |  |  |
| 8 series |  |  |  |  |
| 128 series |  |  |  |  |
| 512 series |  |  |  |  |
| 8 series × (2 past + 2 known) |  |  |  |  |

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
