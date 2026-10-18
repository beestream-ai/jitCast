# Timestral

[← back to the README](../README.md)

jitrex runs [Timestral](https://huggingface.co/mistralai/Timestral-5M), Mistral AI's family of
time-series foundation models (5M, 28M, 123M and 209M parameters). It reproduces
`TimestralForecaster.forecast()` from Mistral's [`timestral`](https://github.com/mistralai/timestral)
package end to end: 21-quantile forecasts for univariate and multivariate series, with past-only and
known-future covariates, any horizon (decoded block by block), and the clampsym test-time
augmentation.

The checkpoints load as Mistral publishes them, a directory with `config.json` and
`model.safetensors`; no conversion is needed.

## Quick start

```bash
# a checkpoint (any size; the weights' licence is below)
mkdir -p Timestral-5M && for f in config.json model.safetensors; do
  curl -L -o Timestral-5M/$f https://huggingface.co/mistralai/Timestral-5M/resolve/main/$f
done
```

```java
try (ForecastEngine engine = TornadoTimestralEngine.load(Path.of("Timestral-5M"))) {
    List<Forecast> forecasts = engine.forecast(List.of(
            TimeSeries.univariate(history),                              // NaN = missing
            new TimeSeries(targets, pastCovariates, futureCovariates)),  // [V][T], [Vp][T], [Vf][T+H]
            64);
    float[] median = forecasts.get(0).median(0);                         // values[target][quantile][step]
}
```

`CpuTimestralEngine` is the same API on the FP32 CPU reference. The forecaster's settings (context
cap, block size, the cap to the observed range, sign-flip averaging, non-negative clipping) are
`TimestralPipeline.Options`, with the reference's defaults:

```java
TimestralPipeline p = engine.pipeline().withOptions(
        TimestralPipeline.Options.defaults(engine.config()).withSymmetricAveraging(false));
```

## How it maps onto jitrex

| Reference | jitrex |
|---|---|
| `TimestralConfig` | `timestral.TimestralConfig` (unknown keys and unsupported settings fail the load) |
| `load_model` | `timestral.TimestralWeights`: every tensor bound by name and shape |
| `TimestralModel.predict_block` | `timestral.BlockForward`: `cpu.CpuTimestral`, `tornado.TimestralForward` |
| masks and positions (`masks.py`) | `timestral.BlockLayout`, shared by both engines |
| `scaler.py` | `timestral.Scaler` |
| `monotone_quantiles` | `timestral.Quantiles` |
| `TimestralForecaster.forecast` | `timestral.TimestralPipeline` |

The scaler (float64 statistics, which Apple GPUs do not have), the patch tokens, the monotone
quantile fan-out and the denormalization (which clamps infinities, which fast-math device compilers
may drop) run on the host, in `timestral.BlockPrep`; they are `O(rows · length)`. The patch
embedding, the transformer and the head's projections run on the device as one task graph per block
shape.

Two weights are rearranged at load time: the fused `q|k|v` projection is stored head-dim-major and
becomes heads-major, and the MLP's fused input projection, whose SiLU half is second, is swapped so
the shared SwiGLU kernel applies.

With covariates the reference also runs a decoder token for every context patch. Those tokens
attend only to themselves and nothing reads them, so jitrex leaves them out; the forecasts are the
same.

The device kernels are listed in [KERNELS.md](KERNELS.md#timestral).

## Validation

`scripts/timestral/dump_golden.py` runs Mistral's forecaster on 13 cases and records, for every
`predict_block` call, its arguments, every stage (scaler, patch embedding, each transformer layer,
the final norm, the head) and its output, plus the final forecasts. It also replays each call in
float64 and records how far each float32 stage is from it: the reference's own noise floor.

| Case | What it covers |
|---|---|
| `univariate_short`, `univariate_refine`, `multi_block` | one decode block; several patches (the refine frame); two blocks with the median fed back (H = 1,200) |
| `multivariate`, `ragged_batch` | three jointly forecast targets; a batch of series of different lengths |
| `with_nans`, `constant`, `nonnegative` | a late start and a gap; a constant series; clipping at 0 |
| `covariates`, `covariates_multi_block`, `multitarget_known` | past and known-future covariates, an indicator covariate over two blocks, two targets with a known covariate |
| `no_symmetric`, `overlong_context` | one pass without sign-flip averaging; a 33,000-step context cut to 32,768 |

| Test | What it checks | Result (Timestral-5M) |
|---|---|---|
| `timestral.cpu.BlockGoldenTest` | every stage of every call of the CPU reference | within 1e-4, or 4× the reference's float32 noise where that is larger |
| `timestral.PipelineGoldenTest` | every call's arguments (bit for bit, fed-back medians within 1e-5) and the forecasts | forecasts within 4e-6 |
| `timestral.tornado.TimestralForwardAccelTest` | the device engine, every call and every forecast | within 4e-5 (blocks), 6e-6 (forecasts) on Metal and OpenCL |
| `timestral.tornado.TimestralBatchParityAccelTest` | a 76-row batch of 37 mixed series against the CPU reference | within 6e-5 |

The noise floor matters in one case: in the 1,024-patch context of `overlong_context` one token's
activations are sensitive enough that the reference's own float32 run is 1e-4 to 4e-4 from its
float64 run in the last layers and the final norm, and two float32 implementations differ by more than 1e-4 there while the
forecasts agree to 1e-6.

```bash
python -m venv venv && . venv/bin/activate && pip install torch safetensors numpy huggingface_hub
git clone https://github.com/mistralai/timestral
python scripts/timestral/dump_golden.py --src timestral/src --ckpt Timestral-5M --out golden-5m

./mvnw test -Dtimestral.checkpoint=Timestral-5M -Dtimestral.golden=golden-5m
./mvnw test -Paccel-tests -Dtest='Timestral*AccelTest' -Dtirex.tornado.device=1:0 \
    -Dtimestral.checkpoint=Timestral-5M -Dtimestral.golden=golden-5m
```

## Performance (Apple M4 Pro, Timestral-5M)

End-to-end `forecast()` with the reference's defaults (sign-flip averaging on), median of repeated
calls after warm-up. jitrex runs on TornadoVM develop (`02e83b2`), Mistral's reference on PyTorch
2.14 on the CPU: its scaler needs float64, which PyTorch's MPS backend does not support, so it
cannot use the GPU. Contexts are 2,048 steps and the horizon 64 unless stated.

| Workload | Reference (CPU) | jitrex Metal | jitrex OpenCL |
|---|---|---|---|
| 1 series | 9.86 ms | **4.70 ms** | 28.97 ms |
| 8 series | 33.87 ms | **12.24 ms** | 36.21 ms |
| 32 series | 68.88 ms | **38.74 ms** | 69.29 ms |
| 128 series | 174.89 ms | **139.79 ms** | 174.96 ms |
| 1 series, H = 1,024 | 11.98 ms | **7.21 ms** | 31.81 ms |
| 8 series, 8,192-step context | 76.48 ms | **44.04 ms** | 106.28 ms |
| 8 series × (2 past + 2 known) covariates | 127.20 ms | **48.32 ms** | 78.43 ms |

```bash
python scripts/timestral/bench_reference.py workloads --out workloads.safetensors
python scripts/timestral/bench_reference.py run --src timestral/src --ckpt Timestral-5M \
    --workloads workloads.safetensors --device cpu --out reference.json
$JITREX ai.beestream.jitrex.bench.TimestralBench --checkpoint Timestral-5M \
    --workloads workloads.safetensors --engine tornado --out tornado.json
```

GEMMs take about 90% of the device time. They run in FP32 on the shared kernels; the FP16 and
cuBLAS GEMM modes of TiRex-2 are not wired to Timestral yet, and it has not been run on CUDA.

## Options

| Property | Default | Effect |
|---|---|---|
| `tirex.tornado.device` | TornadoVM's default | Device, as for TiRex-2 |
| `timestral.tornado.gemm` | `blocked` | `blocked`: the register-blocked FP32 GEMM from `blockedMinRows` rows and the tiled one under it; `tiled`: the tiled one only |
| `timestral.tornado.blockedMinRows` | `128` | The fewest GEMM rows that use the register-blocked kernel |
| `timestral.tornado.plans` | `8` | Block shapes kept compiled (least recently used are evicted) |
| `timestral.tornado.profile` | `false` | Print TornadoVM's per-task kernel times after every execution |
| `timestral.maxTokens` | `131072` | The most tokens one forward pass takes; forecasts do not depend on the split |

## Limits

- Categorical covariates (target-encoded in the reference) are not supported.
- Every checkpoint has 64-wide attention heads; the attention kernels support up to 128, in
  multiples of 8.
- Validated on Timestral-5M. The larger checkpoints share the architecture and load with the same
  code, but have no goldens yet.

## Licence

The Timestral weights are Mistral AI's, under a Modified MIT licence that withholds every right
from companies whose monthly revenue exceeds $20 million; read it before use. Mistral's inference
code is Apache 2.0. jitrex does not distribute the weights.
