# Running jitcast

[← back to the README](../README.md)

Every entry point runs on the TornadoVM JVM with the SDK's argfile:

```bash
JITCAST="java @$TORNADOVM_HOME/tornado-argfile -cp target/classes"
```

## From Java

Every model is a `ForecastEngine`: one call forecasts a batch of series, each with its targets and
optional covariates, and returns quantiles per target.

```java
try (ForecastEngine engine = TornadoTimestralEngine.load(Path.of("Timestral-5M"))) {
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
| [Timestral](models/TIMESTRAL.md) | `timestral.tornado.TornadoTimestralEngine` | `timestral.cpu.CpuTimestralEngine` | a checkpoint directory |
| [Chronos-2](models/CHRONOS2.md) | `chronos2.tornado.TornadoChronos2Engine` | `chronos2.cpu.CpuChronos2Engine` | a checkpoint directory |

- **Quantiles.** TiRex-2: 9 (0.1 … 0.9). Timestral: 21 (0.01 … 0.99). Chronos-2: its checkpoint's
  (21 for `chronos-2`, 13 for `chronos-2-small`).
- **Horizons.** TiRex-2 up to its checkpoint's future length. Timestral and Chronos-2 any horizon.
- **Batches.** A call is split into forward passes by series count and a token budget. A series'
  forecast does not depend on the rest of its batch, except with Chronos-2's cross-learning.
- **Engines** are thread-safe and `AutoCloseable`; close them to free device memory.
- **Options** are JVM properties ([CONFIGURATION.md](CONFIGURATION.md)):

```bash
$JITCAST -Djitcast.tornado.device=1:0 -Dtimestral.tornado.gemm=cublas-fp16 my.App
```

## Benchmark tools

Each model has a tool that times end-to-end `forecast()` on a workload file written by the
reference side in [jitCast-benchmarking](https://github.com/beestream-ai/jitCast-benchmarking),
which also holds the method and results.

```bash
$JITCAST ai.beestream.jitcast.bench.TiRexBench --weights tirex2.safetensors \
    --workloads workloads.safetensors --engine tornado --scenarios throughput,multivariate \
    --batches 1,8,64,512 --out results.json
$JITCAST ai.beestream.jitcast.bench.TimestralBench --checkpoint Timestral-5M \
    --workloads ts-workloads.safetensors --engine tornado --out timestral.json
$JITCAST ai.beestream.jitcast.bench.Chronos2Bench --checkpoint chronos-2 \
    --workloads c2-workloads.safetensors --engine tornado --out chronos2.json
```

| Tool | Options |
|---|---|
| `TiRexBench` | `--engine tornado\|cpu`, `--scenarios`, `--batches`, `--mv-batches`, `--covariates`, `--horizon` (64), `--reps` (30), `--warmup` (5), `--budget` (3 s), `--max-batch` (1024), `--config` |
| `TimestralBench`, `Chronos2Bench` | `--engine tornado\|cpu`, `--scenarios`, `--reps` (20), `--warmup` (3), `--budget` (5 s) |
| `TiRexColdStart` | `--weights`, `--workloads`, `--out`; set `BENCH_EXEC_START_NS` to the launch time |

### Harness servers

`TiRexServer` and `TimestralServer` answer forecast requests over a local socket, so the reference
engines' Python harnesses (fev-bench, GIFT-Eval) can score jitcast unchanged.

```bash
$JITCAST ai.beestream.jitcast.bench.TiRexServer --weights tirex2.safetensors --port-file port.txt
$JITCAST ai.beestream.jitcast.bench.TimestralServer --checkpoint Timestral-5M --port-file port.txt
```

`TiRexServer` takes `--engine tornado|cpu`, `--batch-size` (512), `--plans` (4), `--max-shapes` (12)
and `--max-builds`, after which it asks the client to restart it. `TimestralServer` takes
`--checkpoint`, `--port-file` and `--engine`. Malformed requests get an error and the connection is
closed.

## Tests

The CPU references and host pipelines run against goldens recorded from each reference engine. The
accelerated tests (`*AccelTest`, profile `accel-tests`) check every kernel against its CPU operator
and the device engines against the goldens. Tests without configured goldens are skipped.

```bash
# CPU
./mvnw test -Dtirex.weights=tirex2.safetensors -Dtirex.golden=golden-cpu \
    -Dtimestral.checkpoint=Timestral-5M -Dtimestral.golden=golden-5m \
    -Dchronos2.checkpoint=chronos-2 -Dchronos2.golden=golden-c2
# device
./mvnw test -Paccel-tests -Djitcast.tornado.device=0:0 \
    -Dtirex.weights=tirex2.safetensors -Dtirex.golden=golden-cpu -Dtirex.golden.gpu=golden-cuda \
    -Dtimestral.checkpoint=Timestral-5M -Dtimestral.golden=golden-5m \
    -Dchronos2.checkpoint=chronos-2 -Dchronos2.golden=golden-c2
```

| Model | Golden properties | Recorded by |
|---|---|---|
| TiRex-2 | `tirex.weights`, `tirex.golden`, `tirex.golden.gpu` (reduced-precision checks) | `scripts/dump_golden.py` with NXAI's `tirex-2` v0.4.0, or `scripts/setup_reference.sh` |
| Timestral | `timestral.checkpoint`, `timestral.golden` | `scripts/timestral/dump_golden.py --src <timestral>/src --ckpt <dir> --out <dir>` |
| Chronos-2 | `chronos2.checkpoint`, `chronos2.golden` | `scripts/chronos2/dump_golden.py --src <chronos-forecasting>/src --ckpt <dir> --out <dir>` |

The accelerated profile caps device memory at 8 GB; Timestral-209M with cuBLAS needs
`-Dtornado.device.memory=20GB`.
