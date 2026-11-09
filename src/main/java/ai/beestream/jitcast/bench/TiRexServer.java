/*
 * Copyright (c) 2026 Beestream Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitcast.bench;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.tirex2.ForecastPipeline;
import ai.beestream.jitcast.tirex2.ForwardPass;
import ai.beestream.jitcast.tirex2.TiRex2Weights;
import ai.beestream.jitcast.tirex2.cpu.CpuReference;
import ai.beestream.jitcast.tirex2.tornado.PackedWeights;
import ai.beestream.jitcast.tirex2.tornado.TornadoForward;

/**
 * Serves {@code forecast()} to Python over a local socket, so NXAI's own benchmark harnesses
 * (GIFT-Eval, fev-bench) can score this engine with their data loading and metrics unchanged: the
 * Python side (jitCast-benchmarking's {@code scripts/tirex_tornado.py}) replaces {@code TiRex2.predict} with a call here.
 *
 * <p>Protocol, little-endian, one request at a time:
 * <pre>
 * request : int op (1 = forecast, 0 = quit), int nSeries, int horizon, int signFlip, int trendDiff
 *           (−1 = the checkpoint's default), then per series: int vt, vp, vf, T, Tf and the floats
 *           target [vt·T], past [vp·T], future [vf·Tf]
 * response: int status (0 = ok, else a UTF-8 message: int length, bytes), then per series:
 *           int vt, int nq, int h and the floats [vt·nq·h], then int restart
 * </pre>
 *
 * <p>A plan is built per batch shape and cuBLAS handles leak across plans, so a long
 * benchmark would eventually fail {@code cublasCreate}. Two mitigations: a batch of plain
 * univariate series is padded to a power-of-two size with copies of its first series (their
 * forecasts are dropped), and once {@code --max-shapes} distinct shapes, or {@code --max-builds}
 * plan builds in all, have happened the response asks the client to start a fresh server
 * ({@code restart = 1}). The second limit exists because closing an evicted plan does not give
 * back all of its device memory, so rebuilding the same few shapes eventually runs out.
 *
 * <pre>
 * java … TiRexServer --weights model.safetensors --port-file port.txt [--engine tornado|cpu] [--batch-size 512]
 * </pre>
 */
public final class TiRexServer {

    public static void main(String[] args) throws IOException {
        // The server keeps several plans alive and evicts them as batch shapes change. With
        // pinned host buffers that churn makes TornadoVM's CUDA backend fail device
        // allocations, and pinning does not speed up these workloads; so it is off unless set.
        if (System.getProperty("tornado.cuda.host.pinning") == null) {
            System.setProperty("tornado.cuda.host.pinning", "False");
        }
        Map<String, String> a = TiRexBench.parse(args);
        TiRex2Weights weights = TiRex2Weights.load(Path.of(a.get("weights")));
        String engine = a.getOrDefault("engine", "tornado");
        ForwardPass forward = switch (engine) {
            case "tornado" -> new TornadoForward(weights.config(), PackedWeights.pack(weights),
                    Integer.parseInt(a.getOrDefault("plans", "4")));
            case "cpu" -> new CpuReference(weights);
            default -> throw new IllegalArgumentException("--engine tornado|cpu");
        };
        int batchSize = Integer.parseInt(a.getOrDefault("batch-size", String.valueOf(ForecastPipeline.DEFAULT_BATCH_SIZE)));
        int maxShapes = Integer.parseInt(a.getOrDefault("max-shapes", "12"));
        int maxBuilds = Integer.parseInt(a.getOrDefault("max-builds", "1000000"));
        java.util.Set<TornadoForward.Shape> seen = new java.util.HashSet<>();
        java.util.Set<TornadoForward.Shape> cached = new java.util.HashSet<>();
        int builds = 0;
        ForecastPipeline base = new ForecastPipeline(weights.config(), forward, batchSize, weights.config().ttaDiff(),
                weights.config().ttaSignFlip());

        try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getLoopbackAddress())) {
            Files.writeString(Path.of(a.get("port-file")), String.valueOf(server.getLocalPort()));
            System.err.println("TiRexServer: " + engine + " on port " + server.getLocalPort());
            try (Socket s = server.accept();
                    DataInputStream in = new DataInputStream(new BufferedInputStream(s.getInputStream(), 1 << 20));
                    OutputStream out = new BufferedOutputStream(s.getOutputStream(), 1 << 20)) {
                Wire wire = new Wire(in);
                while (true) {
                    int op = wire.readInt();
                    if (op == 0) {
                        break;
                    }
                    wire.next();
                    int n, horizon, signFlip, trendDiff;
                    List<float[][][]> raw;
                    try {
                        n = wire.count("series count", 1, Wire.MAX_SERIES);
                        horizon = wire.count("horizon", 1, 1 << 20);
                        signFlip = wire.readInt();
                        trendDiff = wire.readInt();
                        raw = new ArrayList<>(n);
                        for (int i = 0; i < n; i++) {
                            int vt = wire.count("targets", 1, 1 << 16), vp = wire.count("past covariates", 0, 1 << 16);
                            int vf = wire.count("known covariates", 0, 1 << 16);
                            int t = wire.count("length", 1, Integer.MAX_VALUE), tf = wire.count("future length", 0, Integer.MAX_VALUE);
                            raw.add(new float[][][] {wire.floats(vt, t), vp == 0 ? null : wire.floats(vp, t),
                                    vf == 0 ? null : wire.floats(vf, tf)});
                        }
                    } catch (Wire.BadRequest e) {
                        // the rest of the request cannot be found: answer and drop the client
                        Wire.error(out, e);
                        out.flush();
                        break;
                    }
                    try {
                        List<TimeSeries> series = new ArrayList<>(n);
                        for (float[][][] r : raw) {
                            series.add(new TimeSeries(r[0], r[1], r[2]));
                        }
                        ForecastPipeline p = new ForecastPipeline(weights.config(), forward, batchSize,
                                trendDiff < 0 ? base.trendDiff() : trendDiff == 1, signFlip < 0 ? base.signFlip() : signFlip == 1);
                        List<Forecast> result = p.forecast(bucketed(series), horizon).subList(0, n);
                        ByteArrayOutputStream body = new ByteArrayOutputStream();
                        for (Forecast f : result) {
                            float[][][] v = f.values();
                            Wire.writeInt(body, v.length);
                            Wire.writeInt(body, v[0].length);
                            Wire.writeInt(body, v[0][0].length);
                            for (float[][] tq : v) {
                                for (float[] q : tq) {
                                    Wire.writeFloats(body, q);
                                }
                            }
                        }
                        if (forward instanceof TornadoForward tf) {
                            List<TornadoForward.Shape> now = tf.cachedShapes();
                            for (TornadoForward.Shape sh : now) {
                                if (!cached.contains(sh)) {
                                    builds++;
                                }
                            }
                            cached = new java.util.HashSet<>(now);
                            seen.addAll(now);
                        }
                        Wire.writeInt(body, seen.size() >= maxShapes || builds >= maxBuilds ? 1 : 0);
                        Wire.ok(out, body);
                    } catch (RuntimeException e) {
                        Wire.error(out, e);
                        e.printStackTrace();
                    }
                    out.flush();
                }
            }
        } finally {
            if (forward instanceof AutoCloseable c) {
                try {
                    c.close();
                } catch (Exception | Error e) {
                    // exiting anyway (after a device OOM TornadoVM can fail to free buffers: TornadoInternalError)
                }
            }
            weights.close();
        }
        System.exit(0);
    }

    /** Pads a batch of plain univariate series to a power-of-two count, so few plan shapes occur. */
    static List<TimeSeries> bucketed(List<TimeSeries> series) {
        boolean plain = series.stream().allMatch(t -> t.targets() == 1 && t.pastCount() == 0 && t.futureCount() == 0);
        int n = series.size();
        int bucket = Integer.highestOneBit(Math.max(1, n - 1)) << 1;
        if (!plain || n == 1 || bucket > ForecastPipeline.DEFAULT_BATCH_SIZE || bucket == n) {
            return series;
        }
        List<TimeSeries> padded = new ArrayList<>(series);
        while (padded.size() < bucket) {
            padded.add(series.get(0));
        }
        return padded;
    }

    private TiRexServer() {}
}
