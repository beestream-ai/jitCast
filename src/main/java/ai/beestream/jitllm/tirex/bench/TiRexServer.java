package ai.beestream.jitllm.tirex.bench;

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

import ai.beestream.jitllm.tirex.api.Forecast;
import ai.beestream.jitllm.tirex.api.TimeSeries;
import ai.beestream.jitllm.tirex.cpu.CpuReference;
import ai.beestream.jitllm.tirex.host.ForecastPipeline;
import ai.beestream.jitllm.tirex.model.ForwardPass;
import ai.beestream.jitllm.tirex.model.TiRex2Weights;
import ai.beestream.jitllm.tirex.tornado.PackedWeights;
import ai.beestream.jitllm.tirex.tornado.TornadoForward;

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
 *           int vt, int nq, int h and the floats [vt·nq·h]
 * </pre>
 *
 * <pre>
 * java … TiRexServer --weights model.safetensors --port-file port.txt [--engine tornado|cpu] [--batch-size 512]
 * </pre>
 */
public final class TiRexServer {

    public static void main(String[] args) throws IOException {
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
                        List<Forecast> result = p.forecast(series, horizon);
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
                } catch (Exception e) {
                    // exiting anyway
                }
            }
            weights.close();
        }
        System.exit(0);
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
