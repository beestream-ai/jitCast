# Running jitcast

[← back to the README](../README.md)

Every entry point runs on the TornadoVM JVM with the SDK's argfile:

```bash
JITCAST="java @$TORNADOVM_HOME/tornado-argfile --add-modules jdk.incubator.vector -cp target/classes"
```

## From Java

Every model is a `ForecastEngine`: one call forecasts a batch of series, each with its targets and
optional covariates, and returns quantiles per target.

```java
try (ForecastEngine engine = TornadoTimestralEngine.load(Path.of("Timestral-5M"))) {   // or TornadoTiRex2Engine
    List<TimeSeries> series = List.of(
            TimeSeries.univariate(values),                                  // float[] history, NaN = missing
            new TimeSeries(targets, pastCovariates, futureCovariates));     // [V][T], [Vp][T], [Vf][T+H]
    List<Forecast> forecasts = engine.forecast(series, 64);                 // horizon 64
    float[] median = forecasts.get(0).median(0);                            // values[target][quantile][step]
    float[] levels = forecasts.get(0).levels();                             // the model's quantile levels
}
```

| Model | TornadoVM engine | CPU reference | Loads |
|---|---|---|---|
| [TiRex-2](models/TIREX2.md) | `tirex2.tornado.TornadoTiRex2Engine` | `tirex2.cpu.CpuTiRex2Engine` | one converted `.safetensors` file |
| [Timestral](models/TIMESTRAL.md) | `timestral.tornado.TornadoTimestralEngine` | `timestral.cpu.CpuTimestralEngine` | a checkpoint directory (`config.json` + `model.safetensors`) |

- **Quantiles.** TiRex-2 gives 9 levels (0.1 … 0.9), Timestral 21 (0.01 … 0.99).
- **Horizons.** TiRex-2 forecasts up to its checkpoint's future length. Timestral forecasts any
  horizon, block by block.
- **Batches.** A forecast call is split into forward passes by series count and by a token budget
  (rows × patches); a series' forecast does not depend on the rest of its batch.
- **Options.** Engine options are JVM properties ([CONFIGURATION.md](CONFIGURATION.md)), for example:

```bash
$JITCAST -Djitcast.tornado.device=1:0 -Dtimestral.tornado.gemm=cublas my.App
```

## Benchmarks

Each model has a benchmark that times end-to-end `forecast()` on the same inputs as its reference
engine; the reference side writes the shared workload file.

```bash
# TiRex-2: latency and throughput (univariate batches, multivariate series × covariates)
python scripts/bench_nxai.py workloads --out workloads.safetensors
$JITCAST -Dtirex.tornado.gemm=cublas-auto ai.beestream.jitcast.bench.TiRexBench \
    --weights tirex2.safetensors --workloads workloads.safetensors --engine tornado \
    --scenarios throughput,multivariate --batches 1,8,64,512 --mv-batches 1,32 --covariates 15 --out results.jsonl

# TiRex-2: cold start (process start → first forecast) and the harness server for NXAI's Python harnesses
BENCH_EXEC_START_NS=$(date +%s%N) $JITCAST ai.beestream.jitcast.bench.TiRexColdStart \
    --weights tirex2.safetensors --workloads workloads.safetensors --out cold.jsonl
$JITCAST ai.beestream.jitcast.bench.TiRexServer --weights tirex2.safetensors --port-file port.txt

# Timestral: 1–512 series, a long horizon, a long context and covariates
python scripts/timestral/bench_reference.py workloads --out ts-workloads.safetensors
$JITCAST ai.beestream.jitcast.bench.TimestralBench --checkpoint Timestral-5M \
    --workloads ts-workloads.safetensors --engine tornado --out timestral.json
```

`TiRexBench` options: `--engine tornado|cpu`, `--scenarios`, `--batches`, `--mv-batches`,
`--covariates`, `--horizon` (64), `--reps` (30), `--warmup` (5), `--budget` (3 s per point),
`--max-batch` (1024), `--config` (a label). `TimestralBench` options: `--engine tornado|cpu`,
`--scenarios`, `--reps` (20), `--warmup` (3), `--budget` (5 s per point). The method and the results
are in [PERFORMANCE.md](PERFORMANCE.md).

## Tests

The CPU references and host pipelines run against goldens recorded from each reference engine; the
accelerated tests (`*AccelTest`, profile `accel-tests`) run every kernel against its CPU operator
and the device engines against the goldens. Tests whose goldens are not configured are skipped.

```bash
# CPU: both models
./mvnw test -Dtirex.weights=tirex2.safetensors -Dtirex.golden=golden-cpu \
    -Dtimestral.checkpoint=Timestral-5M -Dtimestral.golden=golden-5m
# device: kernels and engines (in this profile CUDA is backend 0 on machines that also have OpenCL)
./mvnw test -Paccel-tests -Djitcast.tornado.device=0:0 \
    -Dtirex.weights=tirex2.safetensors -Dtirex.golden=golden-cpu -Dtirex.golden.gpu=golden-cuda \
    -Dtimestral.checkpoint=Timestral-5M -Dtimestral.golden=golden-5m
```

| Model | Goldens | Recorded by |
|---|---|---|
| TiRex-2 | `-Dtirex.weights`, `-Dtirex.golden` (and `-Dtirex.golden.gpu` for the reduced-precision checks) | `scripts/dump_golden.py`, with NXAI's package: `git clone https://github.com/NX-AI/tirex-2 && pip install -e ./tirex-2` |
| Timestral | `-Dtimestral.checkpoint`, `-Dtimestral.golden` | `scripts/timestral/dump_golden.py --src <timestral>/src --ckpt <dir> --out <dir>`, from a clone of `mistralai/timestral` |

The tolerances and what each test checks are in [VALIDATION.md](VALIDATION.md). With cuBLAS, the
Timestral-209M tests need `-Dtornado.device.memory=40GB` (the profile's default is 8 GB).
