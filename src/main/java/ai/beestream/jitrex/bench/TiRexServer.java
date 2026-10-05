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
package ai.beestream.jitrex.bench;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.beestream.jitrex.api.Forecast;
import ai.beestream.jitrex.api.TimeSeries;
import ai.beestream.jitrex.cpu.CpuReference;
import ai.beestream.jitrex.host.ForecastPipeline;
import ai.beestream.jitrex.model.ForwardPass;
import ai.beestream.jitrex.model.TiRex2Weights;
import ai.beestream.jitrex.tornado.PackedWeights;
import ai.beestream.jitrex.tornado.TornadoForward;

/**
 * Serves {@code forecast()} to Python over a local socket, so NXAI's own benchmark harnesses
 * (GIFT-Eval, fev-bench) can score this engine with their data loading and metrics unchanged: the
 * Python side ({@code scripts/tirex_tornado.py}) replaces {@code TiRex2.predict} with a call here.
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
                while (true) {
                    int op = readInt(in);
                    if (op == 0) {
                        break;
                    }
                    int n = readInt(in), horizon = readInt(in), signFlip = readInt(in), trendDiff = readInt(in);
                    List<TimeSeries> series = new ArrayList<>(n);
                    for (int i = 0; i < n; i++) {
                        int vt = readInt(in), vp = readInt(in), vf = readInt(in), t = readInt(in), tf = readInt(in);
                        series.add(new TimeSeries(readRows(in, vt, t), vp == 0 ? null : readRows(in, vp, t),
                                vf == 0 ? null : readRows(in, vf, tf)));
                    }
                    ForecastPipeline p = new ForecastPipeline(weights.config(), forward, batchSize,
                            trendDiff < 0 ? base.trendDiff() : trendDiff == 1, signFlip < 0 ? base.signFlip() : signFlip == 1);
                    try {
                        List<Forecast> result = p.forecast(bucketed(series), horizon).subList(0, n);
                        writeInt(out, 0);
                        for (Forecast f : result) {
                            float[][][] v = f.values();
                            writeInt(out, v.length);
                            writeInt(out, v[0].length);
                            writeInt(out, v[0][0].length);
                            ByteBuffer bb = ByteBuffer.allocate(4 * v.length * v[0].length * v[0][0].length).order(ByteOrder.LITTLE_ENDIAN);
                            for (float[][] tq : v) {
                                for (float[] q : tq) {
                                    bb.asFloatBuffer().put(q);
                                    bb.position(bb.position() + 4 * q.length);
                                }
                            }
                            out.write(bb.array());
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
                        writeInt(out, seen.size() >= maxShapes || builds >= maxBuilds ? 1 : 0);
                    } catch (RuntimeException e) {
                        byte[] msg = String.valueOf(e).getBytes(StandardCharsets.UTF_8);
                        writeInt(out, 1);
                        writeInt(out, msg.length);
                        out.write(msg);
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

    private static int readInt(DataInputStream in) throws IOException {
        return Integer.reverseBytes(in.readInt());
    }

    private static void writeInt(OutputStream out, int v) throws IOException {
        out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array());
    }

    private static float[][] readRows(DataInputStream in, int rows, int cols) throws IOException {
        byte[] bytes = new byte[4 * rows * cols];
        in.readFully(bytes);
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[][] m = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            bb.asFloatBuffer().get(m[r]);
            bb.position(bb.position() + 4 * cols);
        }
        return m;
    }

    private TiRexServer() {}
}
