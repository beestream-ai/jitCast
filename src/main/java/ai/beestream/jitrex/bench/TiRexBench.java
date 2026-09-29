package ai.beestream.jitrex.bench;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitrex.util.Json;
import ai.beestream.jitrex.api.TimeSeries;
import ai.beestream.jitrex.cpu.CpuReference;
import ai.beestream.jitrex.host.ForecastPipeline;
import ai.beestream.jitrex.io.SafeTensors;
import ai.beestream.jitrex.io.Tensor;
import ai.beestream.jitrex.model.ForwardPass;
import ai.beestream.jitrex.model.ModelBatch;
import ai.beestream.jitrex.model.TiRex2Weights;
import ai.beestream.jitrex.tornado.PackedWeights;
import ai.beestream.jitrex.tornado.TornadoForward;

/**
 * The TornadoVM (or CPU) side of the NXAI comparison: the same workload file, scenarios and
 * timing boundaries as {@code scripts/bench_nxai.py}, and the same JSON record layout.
 *
 * <ul>
 *   <li>{@code e2e}: {@code forecast()} — batch assembly, forward, postprocessing;
 *   <li>{@code forward}: the forward passes of that call only (every 512-series window, summed).
 * </ul>
 *
 * <pre>
 * java @$TORNADOVM_HOME/tornado-argfile -cp … ai.beestream.jitrex.bench.TiRexBench \
 *     --weights tirex2.safetensors --workloads workloads.safetensors --engine tornado \
 *     --scenarios throughput,multivariate --out tornado.json
 * </pre>
 */
public final class TiRexBench {

    static int[] BATCHES = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    static int[] COVARIATES = {1, 3, 7, 15, 31, 63};
    static int[] MV_BATCHES = {1, 32};

    /** Records every batch passed to the wrapped forward pass, so it can be replayed alone. */
    static final class Capturing implements ForwardPass {
        final ForwardPass inner;
        final List<ModelBatch> batches = new ArrayList<>();
        boolean capture;

        Capturing(ForwardPass inner) {
            this.inner = inner;
        }

        @Override
        public float[][][] forward(ModelBatch batch) {
            if (capture) {
                batches.add(batch);
            }
            return inner.forward(batch);
        }
    }

    record Scenario(String name, int batch, int covariates, List<TimeSeries> series) {}

    public static void main(String[] args) throws IOException {
        Map<String, String> a = parse(args);
        Path weightsPath = Path.of(a.get("weights"));
        String engine = a.getOrDefault("engine", "tornado");
        int horizon = Integer.parseInt(a.getOrDefault("horizon", "64"));
        int warmup = Integer.parseInt(a.getOrDefault("warmup", "5"));
        int reps = Integer.parseInt(a.getOrDefault("reps", "30"));
        double budget = Double.parseDouble(a.getOrDefault("budget", "3.0"));
        int maxBatch = Integer.parseInt(a.getOrDefault("max-batch", "1024"));
        List<String> which = List.of(a.getOrDefault("scenarios", "throughput,multivariate").split(","));
        if (a.containsKey("mv-batches")) {
            MV_BATCHES = Arrays.stream(a.get("mv-batches").split(",")).mapToInt(Integer::parseInt).toArray();
        }
        if (a.containsKey("covariates")) {
            COVARIATES = Arrays.stream(a.get("covariates").split(",")).mapToInt(Integer::parseInt).toArray();
        }
        if (a.containsKey("batches")) {
            BATCHES = Arrays.stream(a.get("batches").split(",")).mapToInt(Integer::parseInt).toArray();
        }

        long t0 = System.nanoTime();
        TiRex2Weights weights = TiRex2Weights.load(weightsPath);
        ForwardPass fp = switch (engine) {
            case "tornado" -> new TornadoForward(weights.config(), PackedWeights.pack(weights), 1);
            case "cpu" -> new CpuReference(weights);
            default -> throw new IllegalArgumentException("--engine tornado|cpu");
        };
        double loadMs = (System.nanoTime() - t0) / 1e6;
        Capturing cap = new Capturing(fp);
        ForecastPipeline pipeline = new ForecastPipeline(weights.config(), cap);

        List<Map<String, Object>> records = new ArrayList<>();
        try (SafeTensors w = SafeTensors.open(Path.of(a.get("workloads")))) {
            for (Scenario s : scenarios(w, which, maxBatch, horizon)) {
                cap.capture = true;
                long c0 = System.nanoTime();
                pipeline.forecast(s.series(), horizon); // first call: plan build + compile for this shape
                double firstMs = (System.nanoTime() - c0) / 1e6;
                cap.capture = false;
                List<ModelBatch> batches = List.copyOf(cap.batches);
                cap.batches.clear();

                Map<String, Object> e2e = timed(() -> pipeline.forecast(s.series(), horizon), warmup, reps, budget);
                Map<String, Object> fwd = timed(() -> batches.forEach(fp::forward), warmup, reps, budget);
                int rows = batches.stream().mapToInt(ModelBatch::rows).sum();
                double median = (double) e2e.get("median_ms");
                Map<String, Object> rec = new LinkedHashMap<>();
                rec.put("engine", engine);
                rec.put("config", a.getOrDefault("config", engine));
                rec.put("scenario", s.name());
                rec.put("batch", s.batch());
                rec.put("covariates", s.covariates());
                rec.put("rows", rows);
                rec.put("windows", batches.size());
                rec.put("horizon", horizon);
                rec.put("first_call_ms", firstMs);
                rec.put("e2e", e2e);
                rec.put("forward", fwd);
                rec.put("series_per_s", s.batch() / (median / 1e3));
                records.add(rec);
                System.out.printf("%-8s %-12s bs=%5d k=%3d rows=%6d e2e=%9.3f ms  fwd=%9.3f ms  %10.1f series/s  (first %.0f ms)%n",
                        rec.get("config"), s.name(), s.batch(), s.covariates(), rows, median, fwd.get("median_ms"),
                        rec.get("series_per_s"), firstMs);
            }
        }
        Map<String, Object> env = new LinkedHashMap<>();
        env.put("java", System.getProperty("java.version"));
        env.put("device", System.getProperty("tirex.tornado.device", "default"));
        env.put("load_ms", loadMs);
        env.put("argv", String.join(" ", args));
        Map<String, Object> outJson = new LinkedHashMap<>();
        outJson.put("env", env);
        outJson.put("records", records);
        Files.writeString(Path.of(a.get("out")), Json.write(outJson));
        if (fp instanceof AutoCloseable c) {
            try {
                c.close();
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
        }
        weights.close();
    }

    /** The scenarios of bench_nxai.py, from the same workload tensors. */
    static List<Scenario> scenarios(SafeTensors w, List<String> which, int maxBatch, int horizon) {
        List<Scenario> out = new ArrayList<>();
        float[][] uni = rows(w.get("univariate"));
        if (which.contains("latency") || which.contains("throughput")) {
            int[] sizes = which.contains("throughput") ? BATCHES : new int[] {1};
            for (int bs : sizes) {
                if (bs > maxBatch) {
                    continue;
                }
                List<TimeSeries> s = new ArrayList<>();
                for (int i = 0; i < bs; i++) {
                    s.add(TimeSeries.univariate(uni[i]));
                }
                out.add(new Scenario("univariate", bs, 0, s));
            }
        }
        if (which.contains("multivariate")) {
            float[][] target = rows(w.get("mv.target"));
            float[][] fut = rows(w.get("mv.future_cov"));
            float[][] past = rows(w.get("mv.past_cov"));
            // the workload file's own layout (bench_nxai.py's max(COVARIATES)), so a --covariates
            // subset slices exactly the rows NXAI used
            int kMax = Integer.parseInt(w.metadata().getOrDefault("max_covariates", "63"));
            int futPer = (kMax + 1) / 2, pastPer = kMax / 2;
            for (int bs : MV_BATCHES) {
                if (bs > maxBatch) {
                    continue;
                }
                for (int k : COVARIATES) {
                    int nf = (k + 1) / 2, np = k / 2;
                    List<TimeSeries> s = new ArrayList<>();
                    for (int i = 0; i < bs; i++) {
                        float[][] fc = Arrays.copyOfRange(fut, i * futPer, i * futPer + nf);
                        float[][] pc = np == 0 ? null : Arrays.copyOfRange(past, i * pastPer, i * pastPer + np);
                        s.add(new TimeSeries(new float[][] {target[i]}, pc, fc));
                    }
                    out.add(new Scenario("multivariate", bs, k, s));
                }
            }
        }
        return out;
    }

    static float[][] rows(Tensor t) {
        float[] flat = t.toFloatArray();
        int cols = t.dim(1);
        float[][] m = new float[t.dim(0)][];
        for (int r = 0; r < m.length; r++) {
            m[r] = Arrays.copyOfRange(flat, r * cols, (r + 1) * cols);
        }
        return m;
    }

    /** Warm-up, then at least {@code reps} samples and up to {@code budget} seconds (max 10·reps). */
    static Map<String, Object> timed(Runnable fn, int warmup, int reps, double budget) {
        for (int i = 0; i < warmup; i++) {
            fn.run();
        }
        List<Double> t = new ArrayList<>();
        long start = System.nanoTime();
        while (t.size() < reps || ((System.nanoTime() - start) / 1e9 < budget && t.size() < 10 * reps)) {
            long a = System.nanoTime();
            fn.run();
            t.add((System.nanoTime() - a) / 1e6);
            if (t.size() >= reps && (System.nanoTime() - start) / 1e9 > budget) {
                break;
            }
        }
        double[] v = t.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("n", v.length);
        r.put("median_ms", v.length % 2 == 1 ? v[v.length / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2);
        r.put("p10_ms", v[Math.min(v.length - 1, (int) (0.1 * v.length))]);
        r.put("p90_ms", v[Math.min(v.length - 1, (int) (0.9 * v.length))]);
        r.put("min_ms", v[0]);
        r.put("mean_ms", Arrays.stream(v).average().orElse(0));
        List<Object> samples = new ArrayList<>();
        for (double x : v) {
            samples.add(x);
        }
        r.put("samples_ms", samples);
        return r;
    }

    static Map<String, String> parse(String[] args) {
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (!args[i].startsWith("--") || i + 1 >= args.length) {
                throw new IllegalArgumentException("expected --key value pairs, got " + String.join(" ", args));
            }
            m.put(args[i].substring(2), args[++i]);
        }
        return m;
    }

    private TiRexBench() {}
}
