# Chronos-2

[← back to the README](../README.md)

jitcast runs [Chronos-2](https://huggingface.co/amazon/chronos-2), Amazon's time-series foundation
model (`amazon/chronos-2`, 120M parameters, and `autogluon/chronos-2-small`, 28M). It reproduces
`Chronos2Pipeline.predict()` from Amazon's
[`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting) package end to end:
quantile forecasts for univariate and multivariate series, with past-only and known-future
covariates, cross-learning across a batch, and any horizon (past one forward pass, by the
reference's long-horizon unrolling).

The checkpoints load as Amazon publishes them, a directory with `config.json` and
`model.safetensors`; no conversion is needed.

## Quick start

```bash
# a checkpoint (Apache 2.0)
mkdir -p chronos-2 && for f in config.json model.safetensors; do
  curl -L -o chronos-2/$f https://huggingface.co/amazon/chronos-2/resolve/main/$f
done
```

```java
try (ForecastEngine engine = TornadoChronos2Engine.load(Path.of("chronos-2"))) {
    List<Forecast> forecasts = engine.forecast(List.of(
            TimeSeries.univariate(history),                              // NaN = missing
            new TimeSeries(targets, pastCovariates, futureCovariates)),  // [V][T], [Vp][T], [Vf][T+H]
            64);
    float[] median = forecasts.get(0).median(0);                         // values[target][quantile][step]
}
```

`CpuChronos2Engine` is the same API on the FP32 CPU reference. The forecasts carry the checkpoint's
quantile levels: 21 for `chronos-2` (0.01, 0.05, 0.1, …, 0.95, 0.99) and 13 for
`chronos-2-small`. The pipeline's settings (context cap, cross-learning, patches per forward pass,
the unrolled levels) are `Chronos2Pipeline.Options`, with the reference's defaults:

```java
Chronos2Pipeline p = engine.pipeline().withOptions(
        Chronos2Pipeline.Options.defaults(engine.config()).withCrossLearning(true));
```

## How it maps onto jitcast

| Reference | jitcast |
|---|---|
| `Chronos2CoreConfig`, `Chronos2ForecastingConfig` | `chronos2.Chronos2Config` (unknown keys and unsupported settings fail the load) |
| `Chronos2Model` parameters | `chronos2.Chronos2Weights`: every tensor bound by name and shape |
| `Chronos2Model.forward` | `chronos2.ModelForward`: `cpu.CpuChronos2`, `tornado.Chronos2Forward` |
| masks and positions (`Chronos2Encoder`) | `chronos2.ModelLayout`, shared by both engines |
| `InstanceNorm`, patching, time encoding, inverse | `chronos2.ModelPrep` |
| `interpolate_quantiles`, `weighted_quantile` | `chronos2.QuantileMath` |
| `Chronos2Pipeline.predict`, `Chronos2Dataset` | `chronos2.Chronos2Pipeline` |

A forward pass is a T5-style encoder over each row's tokens: one per 16-step context patch, the
`[REG]` token, and one per forecast patch. Each block runs time attention along a row (RoPE,
bidirectional, over the tokens with an observation), group attention across the rows of a series
at each token (the targets and covariates of one series, or the whole batch with cross-learning),
and a ReLU feed-forward layer. As in the reference, attention scores are not scaled by `1/√d_kv`.

The instance norm (a row's mean and standard deviation, then arcsinh), the patch tokens (time
encoding, scaled values, observation mask) and the inverse norm run on the host, in
`chronos2.ModelPrep`; they are `O(rows · length)`. The patch embedding, the encoder and the output
patch embedding run on the device as one task graph per shape (rows, context patches, output
patches). The device engine reorders the rows so that each group is a contiguous run, which the
group-attention kernel needs, and restores the order after.

Horizons past `max_output_patches · 16` steps (1,024) unroll as the reference does: every row is
repeated once per unrolled level (0.1 … 0.9), each copy's context is extended with that level of
the forecast so far (and the known covariates with their real future), and the next pass's
forecasts of all copies are merged back into the quantile levels as a weighted sample. A pass
larger than `-Dchronos2.maxTokens` tokens runs as several, split between groups, with the same
forecasts.

The device kernels are listed in [KERNELS.md](KERNELS.md#chronos-2).

## Validation

`scripts/chronos2/dump_golden.py` runs Amazon's pipeline on 13 cases and records, for every
`Chronos2Model.forward` call, its arguments, every stage (the instance norm, the context and future
patch embeddings, each encoder block, the final norm, the output patch embedding) and its output,
plus the final forecasts. It also replays each call in float64 and records how far each float32
stage is from it: the reference's own noise floor.

| Case | What it covers |
|---|---|
| `univariate_short`, `univariate_long` | one forward pass of 1 and 32 output patches |
| `multi_step`, `covariates_multi_step` | horizons of 1,300 and 1,100 steps: a pass of 64 patches, then the unrolled pass (the second with a known indicator covariate sliding into the context) |
| `multivariate`, `multitarget_known` | three jointly forecast targets; two targets with a known covariate |
| `with_nans`, `constant`, `ragged_batch` | a late start and a gap; a constant series; series of different lengths, left-padded |
| `covariates` | two past-only and two known covariates |
| `long_context` | a 9,000-step context cut to 8,192 |
| `cross_learning`, `many_series` | four series in one group; 300 series in two batches of the default 256 rows |

| Test | What it checks |
|---|---|
| `chronos2.cpu.ForwardGoldenTest` | every stage of every call of the CPU reference: within 1e-4, or 4× the reference's float32 noise where that is larger |
| `chronos2.PipelineGoldenTest` | every call's arguments (bit for bit; unrolled contexts within 1e-5) and the forecasts (within 1e-4); a split forward pass gives the same forecasts |
| `chronos2.tornado.Chronos2ForwardAccelTest` | the device engine, every call and every forecast (within 1e-4) |
| `chronos2.tornado.Chronos2BatchParityAccelTest` | a mixed batch of 25 series (ragged, gaps, late starts, multivariate, covariates, unrolled) against the CPU reference (within 1e-4) |

The encoder's activations are compared on the tokens that have an observation. A context patch
that no row of its group observes is never read by any other token; for it the reference's group
attention averages every row of the batch (its additive mask swamps the scores) and jitcast writes
zeros, so those tokens differ while the forecasts do not.

Worst relative errors (TornadoVM develop, Apple M4 Pro):

| Checkpoint | CPU stages | CPU forecasts | Metal calls / forecasts | OpenCL calls / forecasts | Mixed batch (Metal / OpenCL) |
|---|---|---|---|---|---|
| chronos-2-small | 5.2e-5 | 5.5e-6 | 3.4e-6 / 5.5e-6 | 2.5e-6 / 5.4e-6 | 8.4e-6 / 8.0e-6 |
| chronos-2 | 4.4e-5 | 4.5e-5 | 3.3e-6 / 4.5e-5 | 3.0e-6 / 4.5e-5 | 1.3e-5 / 1.1e-5 |

The worst stage is in `constant`, whose rows have the floor scale (1e-5), where the reference's own
float32 noise is of the same size. The forecasts' worst case is the unrolled `multi_step` on
`chronos-2`: merging 189 weighted samples per step sorts them, and samples that differ by float32
rounding can swap places. Every forecast that does not unroll is within 3.5e-6.

```bash
python -m venv venv && . venv/bin/activate && pip install torch "transformers>=4.41,<6" einops pandas safetensors
git clone https://github.com/amazon-science/chronos-forecasting
python scripts/chronos2/dump_golden.py --src chronos-forecasting/src --ckpt chronos-2 --out golden-c2

./mvnw test -Dtest='ForwardGoldenTest,PipelineGoldenTest' -Dchronos2.checkpoint=chronos-2 -Dchronos2.golden=golden-c2
./mvnw test -Paccel-tests -Dtest='Chronos2*AccelTest' -Djitcast.tornado.device=1:0 \
    -Dchronos2.checkpoint=chronos-2 -Dchronos2.golden=golden-c2
```

## Performance

End-to-end `predict()` / `forecast()`, the median of repeated calls after warm-up, on the same
inputs for both engines (`bench_reference.py workloads`: contexts of 2,048 steps and a horizon of 64
unless stated). jitcast runs on TornadoVM develop; Amazon's reference on PyTorch 2.14 in eager
mode. Speed-ups are over the reference.

### Apple M4 Pro

jitcast on Metal with its default GEMMs (register-blocked FP32), against the reference on the CPU
and on MPS (its forecasts there match the CPU run within 9e-6), in ms; the fastest of each row in
bold. Measured before the time attention took blocks of keys:

| Workload | small ref. CPU | small ref. MPS | small jitcast Metal | base ref. CPU | base ref. MPS | base jitcast Metal |
|---|---|---|---|---|---|---|
| 1 series | 16.2 | **8.3** | 8.5 | 45.1 | **19.8** | 30.8 |
| 8 series | 39.9 | **31.5** | 34.3 | 129.3 | **103.2** | 157.9 |
| 32 series | 115.6 | **81.3** | 121.6 | 414.5 | **302.5** | 552.9 |
| 128 series | 457.6 | **328.0** | 476.2 | 1,720.4 | **1,202.8** | 2,166.6 |
| 512 series | 1,942.7 | **1,514.5** | 1,878.3 | 7,133.4 | **5,494.1** | 8,530.0 |
| 1 series, H = 1,024 | 18.1 | **10.0** | 10.2 | 48.3 | **25.8** | 39.4 |
| 1 series, H = 2,048 (unrolled) | 91.0 | **66.8** | 90.0 | 308.9 | **222.4** | 381.0 |
| 8 series, 8,192-step context | 127.3 | **106.2** | 149.0 | 479.3 | **387.4** | 644.4 |
| 8 series × (2 past + 2 known) | 143.8 | **101.9** | 150.5 | 547.9 | **379.4** | 688.8 |

On Apple silicon the reference on MPS is faster than jitcast on every workload, by 1.0–1.5× on
`chronos-2-small` and 1.5–1.8× on `chronos-2`. TornadoVM's profiler
(`-Dchronos2.tornado.profile=true`) puts about 90% of jitcast's device time in the GEMMs and 6% in
the attention kernels, with the device busy for about 94% of the wall time (`chronos-2`, 32
series). The register-blocked FP32 GEMMs reach about 2 TFLOP/s on these shapes (2.15 for the
64 × 128 tiles of `gemmBlockedWide`, from 1,024 rows), against about 3.2 TFLOP/s for the whole forward pass on MPS, whose GEMMs can use Apple's simdgroup matrix
instructions; jitcast's kernels use scalar FMAs.

### NVIDIA GeForce RTX 5070 Ti

jitcast on CUDA with its default GEMMs (cuBLAS FP32, CUDA graphs) and with cuBLAS FP16, against the
reference on PyTorch CUDA (FP32), in ms, with the speed-up over the reference (TornadoVM develop,
CUDA 13.0, driver 580; jitcast's forecasts are within 4.5e-5 of the reference in FP32 and 5.1e-3
in FP16):

| Workload | small ref. | small jitcast FP32 | small jitcast FP16 | base ref. | base jitcast FP32 | base jitcast FP16 |
|---|---|---|---|---|---|---|
| 1 series | 5.34 | **1.38 (3.87×)** | 0.75 (7.14×) | 8.70 | **4.41 (1.97×)** | 1.72 (5.06×) |
| 8 series | 5.77 | **4.09 (1.41×)** | 2.15 (2.68×) | 18.22 | **15.02 (1.21×)** | 6.65 (2.74×) |
| 32 series | 15.49 | **13.64 (1.14×)** | 7.47 (2.07×) | 59.77 | **56.60 (1.06×)** | 24.97 (2.39×) |
| 128 series | 68.92 | **53.42 (1.29×)** | 33.01 (2.09×) | 272.62 | **232.52 (1.17×)** | 106.73 (2.55×) |
| 512 series | 308.26 | **220.93 (1.40×)** | 137.03 (2.25×) | 1,127.57 | **890.12 (1.27×)** | 415.28 (2.72×) |
| 1 series, H = 1,024 | 5.11 | **1.59 (3.22×)** | 0.96 (5.30×) | 8.94 | **5.75 (1.56×)** | 2.37 (3.77×) |
| 1 series, H = 2,048 (unrolled) | 15.36 | **14.58 (1.05×)** | 12.21 (1.26×) | **43.83** | 44.10 (0.99×) | 24.69 (1.78×) |
| 8 series, 8,192-step context | **15.77** | 16.90 (0.93×) | 10.84 (1.45×) | **62.81** | 67.20 (0.93×) | 35.86 (1.75×) |
| 8 series × (2 past + 2 known) | 19.57 | **17.41 (1.12×)** | 9.44 (2.07×) | 75.58 | **71.50 (1.06×)** | 33.21 (2.28×) |

- In FP32 jitcast is fastest on small batches and short contexts (up to 3.9×), where the
  reference's time is launch and Python overhead, and ahead of it by 1.05–1.4× up to 512 series.
- With long token sequences the two are within 10%: the reference is ahead on the 8,192-step
  context (517 tokens per row) and level on the unrolled horizon. There the time attention is 28% of jitcast's device time (`chronos-2`), and the GEMMs most of the
  rest.
- FP16 GEMMs (`-Dchronos2.tornado.gemm=cublas-fp16`) make jitcast faster than the reference on
  every workload, by 1.3–7.1×; the default is FP32.

### Reproducing

```bash
python scripts/chronos2/bench_reference.py workloads --out workloads.safetensors
python scripts/chronos2/bench_reference.py run --src chronos-forecasting/src --ckpt chronos-2 \
    --workloads workloads.safetensors --device mps --out reference.json        # or cpu, cuda
$JITCAST -Djitcast.tornado.device=1:0 ai.beestream.jitcast.bench.Chronos2Bench --checkpoint chronos-2 \
    --workloads workloads.safetensors --engine tornado --out jitcast.json
# on CUDA (see INSTALL.md#nvidia-extras for cuBLAS 13 on Blackwell)
$JITCAST -Djitcast.tornado.device=1:0 -Dchronos2.tornado.gemm=cublas-fp16 ai.beestream.jitcast.bench.Chronos2Bench ...
```

## Options

| Property | Default | Effect |
|---|---|---|
| `jitcast.tornado.device` | TornadoVM's default | Device, as for TiRex-2 |
| `chronos2.tornado.gemm` | `auto` | `auto`: `cublas` on CUDA, `blocked` elsewhere; the other modes as for Timestral ([TIMESTRAL.md](TIMESTRAL.md#options)) |
| `chronos2.tornado.cudaGraph` | `true` with cuBLAS | Capture the task graph once and replay it as a CUDA graph |
| `chronos2.tornado.blockedMinRows` | `128` | The fewest GEMM rows that use the register-blocked kernel |
| `chronos2.tornado.wideMinRows` | `1024` | The fewest GEMM rows that use the register-blocked kernel with 64 × 128 tiles |
| `chronos2.tornado.plans` | `8` | Shapes kept compiled (least recently used are evicted) |
| `chronos2.tornado.profile` | `false` | Print TornadoVM's per-task kernel times after every execution |
| `chronos2.maxTokens` | `131072` | The most tokens one forward pass takes; forecasts do not depend on the split |

## Limits

- Categorical covariates (target-encoded in the reference) are not supported; encode them as
  numbers first.
- Only float32 checkpoints with the ReLU feed-forward layer, non-overlapping patches and equal input
  and output patch sizes load; every published Chronos-2 checkpoint is one. On the device, heads
  must be 16 to 64 wide, in multiples of 16 (all are 64).
- Validated on `chronos-2` and `chronos-2-small`. `autogluon/chronos-2-synth` shares their
  architecture and loads with the same code, but has no goldens.

## Licence

The Chronos-2 weights and Amazon's inference code are under the Apache License 2.0. jitcast does
not distribute the weights.
