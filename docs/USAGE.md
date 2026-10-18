# Running jitrex

[← back to the README](../README.md)

Every entry point runs on the TornadoVM JVM with the SDK's argfile:

```bash
JITREX="java @$TORNADOVM_HOME/tornado-argfile --add-modules jdk.incubator.vector -cp target/classes"
```

## From Java

```java
try (ForecastEngine engine = TornadoTiRex2Engine.load(Path.of("tirex2.safetensors"))) {
    List<TimeSeries> series = List.of(
            TimeSeries.univariate(values),                        // float[] history, NaN = missing
            new TimeSeries(targets, pastCovariates, futureCovariates));  // [V][T], [Vp][T], [Vf][T+H]
    List<Forecast> forecasts = engine.forecast(series, 64);       // horizon 64
    float[] median = forecasts.get(0).median(0);                  // values[target][quantile][step]
}
```

`CpuTiRex2Engine` is the same API on the CPU reference. Engine options are JVM properties (`-Dtirex.tornado.*`, see [CONFIGURATION.md](CONFIGURATION.md)), for example:

```bash
$JITREX -Dtirex.tornado.device=0:0 -Dtirex.tornado.gemm=cublas-auto my.App
```

## Benchmark, cold start and harness server

```bash
# the shared benchmark inputs (the same file drives the NXAI baseline)
python scripts/bench_nxai.py workloads --out workloads.safetensors

# latency and throughput: univariate batches and multivariate series × covariates
$JITREX -Dtirex.tornado.gemm=cublas-auto ai.beestream.jitrex.bench.TiRexBench \
    --weights tirex2.safetensors --workloads workloads.safetensors --engine tornado \
    --scenarios throughput,multivariate --batches 1,8,64,512 --mv-batches 1,32 --covariates 15 --out results.jsonl

# cold start: process start → first forecast, split into runtime init, load, first and second forecast
BENCH_EXEC_START_NS=$(date +%s%N) $JITREX ai.beestream.jitrex.bench.TiRexColdStart \
    --weights tirex2.safetensors --workloads workloads.safetensors --out cold.jsonl

# forecasts over a local socket, for NXAI's Python harnesses (scripts/tirex_tornado.py is the client)
$JITREX ai.beestream.jitrex.bench.TiRexServer --weights tirex2.safetensors --port-file port.txt
```

`TiRexBench` options: `--engine tornado|cpu`, `--scenarios`, `--batches`, `--mv-batches`, `--covariates`, `--horizon` (64), `--reps` (30), `--warmup` (5), `--budget` (3 s per point), `--max-batch` (1024), `--config` (a label).

Timestral's engines, benchmark and tests are in [TIMESTRAL.md](TIMESTRAL.md).

## Tests

```bash
# CPU reference and host pipeline against the goldens (skipped unless both paths are set)
./mvnw test -Dtirex.weights=tirex2.safetensors -Dtirex.golden=golden-cpu
# device: each kernel against the CPU operator, and the whole engine against the goldens
./mvnw test -Paccel-tests -Dtest='*AccelTest' -Dtirex.tornado.device=1:0 \
    -Dtirex.weights=tirex2.safetensors -Dtirex.golden=golden-cpu -Dtirex.golden.gpu=golden-cuda
```

FP32 configurations must stay within 1e-4 of the FP32 goldens. Configurations that use FP16 or TF32 anywhere may be, per case, no further from them than NXAI's own GPU path is: `-Dtirex.golden.gpu` points at NXAI's CUDA goldens (`dump_golden.py --device cuda`, recorded on the GPU under test), and without it those checks are skipped. Record both golden sets on the same machine: `dump_golden.py` generates the inputs on the host, and their last bits differ between CPU architectures; the tests check that the two sets have identical inputs.

The goldens come from `scripts/dump_golden.py`, (see [VALIDATION.md](VALIDATION.md)), which needs NXAI's package: `git clone https://github.com/NX-AI/tirex-2 && pip install -e ./tirex-2`.
