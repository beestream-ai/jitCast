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

| Test | What it checks |
|---|---|
| `timestral.cpu.BlockGoldenTest` | every stage of every call of the CPU reference: within 1e-4, or 4× the reference's float32 noise where that is larger |
| `timestral.PipelineGoldenTest` | every call's arguments (bit for bit, fed-back medians within 1e-5) and the forecasts (within 1e-4) |
| `timestral.tornado.TimestralForwardAccelTest` | the device engine, every call and every forecast (within 1e-4) |
| `timestral.tornado.TimestralBatchParityAccelTest` | a 76-row batch of 37 mixed series against the CPU reference (within 1e-4) |

Worst relative errors, with goldens recorded from each checkpoint (TornadoVM develop `02e83b2`, Apple M4 Pro):

| Checkpoint | CPU blocks | CPU forecasts | Metal blocks / forecasts | OpenCL blocks / forecasts | Mixed batch (Metal / OpenCL) |
|---|---|---|---|---|---|
| Timestral-5M | 4.7e-5 | 4.0e-6 | 3.0e-5 / 6.2e-6 | 3.8e-5 / 2.2e-6 | 5.3e-5 / 5.0e-5 |
| Timestral-28M | 1.3e-5 | 2.7e-6 | 1.6e-5 / 7.5e-6 | 8.5e-6 / 2.9e-6 | 4.6e-5 / 4.1e-5 |
| Timestral-209M | 6.4e-5 | 1.4e-6 | 6.3e-5 / 2.0e-6 | 6.5e-5 / 4.2e-6 | 1.4e-5 / — |

On the A100 (CUDA and OpenCL, TornadoVM's own FP32 GEMMs, and cuBLAS FP32) every test passes on the
three checkpoints, within 6e-5 for blocks and 1.2e-5 for forecasts. Under `-Paccel-tests` TornadoVM
lists CUDA first (`tornado.cuda.priority`), so its CUDA device is `0:0` in the tests and `1:0` in a
plain launch on a machine whose `tornado --devices` lists OpenCL first.

### Accuracy of the FP16 modes

With FP16 GEMM inputs (`half`, `cublas-fp16`) the forecasts move by up to 1e-2 of the series' range
on Timestral-5M (4e-2 in one long two-block horizon), 2e-3 on 28M and 4e-3 on 209M; the tests
check those modes against a 5e-2 bound only. The default GEMMs are FP32.

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

## Performance

End-to-end `forecast()` with the reference's defaults (sign-flip averaging on), the median of
repeated calls after warm-up, on the same inputs for both engines (`bench_reference.py workloads`:
contexts of 2,048 steps and a horizon of 64 unless stated). jitrex runs on TornadoVM develop
(`02e83b2`), Mistral's reference on PyTorch 2.14 in eager mode. Speed-ups are over the reference.

### NVIDIA A100 80GB PCIe

jitrex with its default GEMMs on CUDA (cuBLAS FP32, CUDA graphs) and with cuBLAS FP16, against the
reference on PyTorch CUDA (FP32), in ms:

| Workload | 5M reference | 5M jitrex FP32 | 5M jitrex FP16 | 28M reference | 28M jitrex FP32 | 28M jitrex FP16 | 209M reference | 209M jitrex FP32 | 209M jitrex FP16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 series | 44.4 | **4.7 (9.4×)** | 4.4 (10.0×) | 57.4 | **6.4 (9.0×)** | 5.8 (9.9×) | 90.6 | **21.9 (4.1×)** | 9.3 (9.8×) |
| 8 series | 46.3 | **12.1 (3.8×)** | 9.9 (4.7×) | 60.0 | **16.1 (3.7×)** | 12.7 (4.7×) | 94.9 | **45.3 (2.1×)** | 21.7 (4.4×) |
| 32 series | 46.2 | **21.4 (2.2×)** | 19.9 (2.3×) | 60.2 | **35.5 (1.7×)** | 26.2 (2.3×) | 147.6 | **134.1 (1.1×)** | 52.3 (2.8×) |
| 128 series | **46.9** | 79.2 (0.59×) | 73.1 (0.64×) | **95.9** | 130.0 (0.74×) | 90.1 (1.06×) | 512.9 | 526.6 (0.97×) | 179.8 (2.9×) |
| 512 series | **102.4** | 308.7 (0.33×) | 288.6 (0.35×) | **349.6** | 512.0 (0.68×) | 353.9 (0.99×) | 2,067.9 | 2,018.1 (1.02×) | 700.6 (3.0×) |
| 1 series, H = 1,024 | 78.6 | **7.8 (10.1×)** | 8.8 (8.9×) | 91.2 | **9.7 (9.4×)** | 9.0 (10.2×) | 124.1 | **28.1 (4.4×)** | 13.4 (9.2×) |
| 8 series, 8,192-step context | 47.9 | **29.9 (1.6×)** | 28.1 (1.7×) | 61.9 | **38.4 (1.6×)** | 37.7 (1.6×) | 153.4 | **146.3 (1.05×)** | 67.7 (2.3×) |
| 8 series × (2 past + 2 known) | 51.7 | **24.7 (2.1×)** | 22.3 (2.3×) | 77.2 | **42.5 (1.8×)** | 29.3 (2.6×) | 336.3 | **174.5 (1.9×)** | 61.2 (5.5×) |

- The reference costs 45–90 ms per call whatever the batch up to 32 series: launch and Python
  overhead, which jitrex does not have. Its time grows only once its GEMMs fill the GPU.
- At large batches of the small checkpoints the reference is faster: there jitrex's time is not in
  its GEMMs but in the rest of a block (its attention kernels and the host steps), still to profile.
- FP16 GEMMs change the forecasts (see *Accuracy of the FP16 modes*); FP32 is the default.

### Apple M4 Pro

The reference's scaler and refine step compute in float64, which PyTorch's MPS backend does not
support, so as published it runs on the CPU only. `bench_reference.py --device mps` runs those two
steps on the CPU and the rest on the GPU (its forecasts match the CPU run to 8e-6). jitrex runs on
Metal with its default GEMMs (register-blocked FP32). In ms, with the speed-up over the faster of
the reference's CPU and MPS runs:

| Workload | 5M ref. CPU | 5M ref. MPS | 5M jitrex Metal | 28M ref. CPU | 28M ref. MPS | 28M jitrex Metal | 209M ref. CPU | 209M ref. MPS | 209M jitrex Metal |
|---|---|---|---|---|---|---|---|---|---|
| 1 series | 10.0 | 14.9 | **5.1 (2.0×)** | 22.8 | 22.8 | **18.0 (1.3×)** | 108.0 | **53.6** | 113.8 (0.47×) |
| 8 series | 33.6 | 23.1 | **12.3 (1.9×)** | 73.9 | **44.4** | 49.1 (0.90×) | 272.6 | **173.2** | 324.5 (0.53×) |
| 32 series | 68.9 | 47.6 | **39.7 (1.2×)** | 176.3 | **110.2** | 164.2 (0.67×) | 861.0 | **531.7** | 1,156.9 (0.46×) |
| 128 series | 184.7 | **121.1** | 139.8 (0.87×) | 560.5 | **388.6** | 627.9 (0.62×) | 3,287.0 | **2,201.9** | 4,577.1 (0.48×) |
| 512 series | 809.1 | 625.6 | **592.0 (1.06×)** | 2,551.8 | **1,939.4** | 2,494.2 (0.78×) | 12,830.4 | **10,372.6** | 18,226.2 (0.57×) |
| 1 series, H = 1,024 | 12.0 | 17.9 | **6.7 (1.8×)** | 41.2 | 25.7 | **22.0 (1.2×)** | 125.3 | **55.7** | 139.5 (0.40×) |
| 8 series, 8,192-step context | 72.9 | 48.9 | **41.0 (1.2×)** | 179.8 | **119.2** | 171.5 (0.69×) | 854.9 | **557.0** | 1,194.0 (0.47×) |
| 8 series × covariates | 123.8 | 85.7 | **48.4 (1.8×)** | 384.9 | 250.9 | **210.7 (1.2×)** | 1,944.6 | **1,363.6** | 1,500.7 (0.91×) |

On Metal about 95% of jitrex's device time is in its GEMMs, which reach about half the throughput of
the matrix multiplications PyTorch's MPS backend calls; FP16 (`half`) does not change that. Closing
the gap needs a GEMM on Apple's matrix hardware, which TornadoVM's Metal backend does not expose to
jitrex today.

### Reproducing

```bash
python scripts/timestral/bench_reference.py workloads --out workloads.safetensors
python scripts/timestral/bench_reference.py run --src timestral/src --ckpt Timestral-5M \
    --workloads workloads.safetensors --device cuda --out reference.json        # or cpu, mps
$JITREX -Dtirex.tornado.device=1:0 ai.beestream.jitrex.bench.TimestralBench --checkpoint Timestral-5M \
    --workloads workloads.safetensors --engine tornado --out jitrex.json
scripts/timestral/eval_mac.sh <workdir> timestral/src Timestral-5M Timestral-28M Timestral-209M   # the Mac matrix
```

## Options

| Property | Default | Effect |
|---|---|---|
| `tirex.tornado.device` | TornadoVM's default | Device, as for TiRex-2 |
| `timestral.tornado.gemm` | `auto` | `auto`: `cublas` on CUDA, `blocked` elsewhere. `blocked`: TornadoVM's register-blocked FP32 GEMM from `blockedMinRows` rows and the tiled one under it; `tiled`: the tiled one only; `half`: TornadoVM's FP16 GEMM (FP32 accumulation); `cublas`, `cublas-tf32`, `cublas-fp16`: cuBLAS in FP32, TF32, or FP16 with FP32 accumulation (CUDA) |
| `timestral.tornado.cudaGraph` | `true` with cuBLAS | Capture the task graph once and replay it as a CUDA graph |
| `timestral.tornado.blockedMinRows` | `128` | The fewest GEMM rows that use the register-blocked kernel |
| `timestral.tornado.plans` | `8` | Block shapes kept compiled (least recently used are evicted) |
| `timestral.tornado.profile` | `false` | Print TornadoVM's per-task kernel times after every execution |
| `timestral.maxTokens` | `131072` | The most tokens one forward pass takes; forecasts do not depend on the split |

## Limits

- Categorical covariates (target-encoded in the reference) are not supported.
- Every checkpoint has 64-wide attention heads; the attention kernels support up to 128, in
  multiples of 8.
- Validated on Timestral-5M, 28M and 209M. Timestral-123M shares their architecture and loads with
  the same code, but has no goldens.

## Licence

The Timestral weights are Mistral AI's, under a Modified MIT licence that withholds every right
from companies whose monthly revenue exceeds $20 million; read it before use. Mistral's inference
code is Apache 2.0. jitrex does not distribute the weights.
