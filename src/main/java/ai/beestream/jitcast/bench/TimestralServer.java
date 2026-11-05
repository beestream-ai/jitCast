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
import java.io.DataInputStream;
import java.io.EOFException;
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

import ai.beestream.jitcast.api.Forecast;
import ai.beestream.jitcast.api.ForecastEngine;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.timestral.Timing;
import ai.beestream.jitcast.timestral.cpu.CpuTimestralEngine;
import ai.beestream.jitcast.timestral.tornado.TornadoTimestralEngine;

/**
 * Serves Timestral's {@code forecast()} to Python over a local socket, so benchmark harnesses
 * ({@code scripts/timestral/fev_eval.py}) can score jitcast with their own data loading and metrics.
 *
 * <p>Protocol, little-endian, one request at a time:
 * <pre>
 * request : int op (1 = forecast, 0 = quit the server), int nSeries, int horizon, then per series:
 *           int vt, vp, vf, cp, cf, T, Tf, Tcf; the floats target [vt·T], past [vp·T], future [vf·Tf];
 *           the doubles categorical past [cp·T], categorical future [cf·Tcf]
 * response: int status (0 = ok, else a UTF-8 message: int length, bytes), int nq, the floats of the
 *           quantile levels [nq], then per series: int vt, int h and the floats [vt·nq·h]
 * </pre>
 *
 * <p>Clients are served one after another: one that disconnects without quitting leaves the server,
 * with its compiled kernels and plans, waiting for the next.
 *
 * <pre>
 * java … TimestralServer --checkpoint Timestral-5M --port-file port.txt [--engine tornado|cpu]
 * </pre>
 */
public final class TimestralServer {

    private TimestralServer() {}

    public static void main(String[] args) throws IOException {
        if (System.getProperty("tornado.cuda.host.pinning") == null) {
            System.setProperty("tornado.cuda.host.pinning", "False");
        }
        Map<String, String> a = TiRexBench.parse(args);
        Path checkpoint = Path.of(a.get("checkpoint"));
        String engineName = a.getOrDefault("engine", "tornado");
        try (ForecastEngine engine = engineName.equals("cpu") ? CpuTimestralEngine.load(checkpoint)
                : TornadoTimestralEngine.load(checkpoint);
                ServerSocket server = new ServerSocket(0, 1, InetAddress.getLoopbackAddress())) {
            Files.writeString(Path.of(a.get("port-file")), String.valueOf(server.getLocalPort()));
            System.err.println("TimestralServer: " + engine.name() + " on port " + server.getLocalPort());
            boolean quit = false;
            while (!quit) {
                try (Socket s = server.accept();
                        DataInputStream in = new DataInputStream(new BufferedInputStream(s.getInputStream(), 1 << 20));
                        OutputStream out = new BufferedOutputStream(s.getOutputStream(), 1 << 20)) {
                    quit = serve(engine, in, out);
                }
            }
        }
        System.exit(0);
    }

    /** Answers one client's requests: true when it asked the server to quit, false when it disconnected. */
    private static boolean serve(ForecastEngine engine, DataInputStream in, OutputStream out) throws IOException {
        while (true) {
            int op;
            try {
                op = readInt(in);
            } catch (EOFException e) {
                return false;
            }
            if (op == 0) {
                return true;
            }
            int n = readInt(in), horizon = readInt(in);
            List<TimeSeries> series = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                int vt = readInt(in), vp = readInt(in), vf = readInt(in), cp = readInt(in), cf = readInt(in);
                int t = readInt(in), tf = readInt(in), tcf = readInt(in);
                float[][] target = readFloats(in, vt, t);
                float[][] past = vp == 0 ? null : readFloats(in, vp, t);
                float[][] future = vf == 0 ? null : readFloats(in, vf, tf);
                double[][] catPast = cp == 0 ? null : readDoubles(in, cp, t);
                double[][] catFuture = cf == 0 ? null : readDoubles(in, cf, tcf);
                series.add(new TimeSeries(target, past, future, catPast, catFuture));
            }
            try {
                List<Forecast> result = engine.forecast(series, horizon);
                Timing.report("request of " + n + " series", 1);
                writeInt(out, 0);
                float[] levels = result.get(0).levels();
                writeInt(out, levels.length);
                out.write(floats(levels));
                for (Forecast f : result) {
                    float[][][] v = f.values();
                    writeInt(out, v.length);
                    writeInt(out, v[0][0].length);
                    for (float[][] tq : v) {
                        for (float[] q : tq) {
                            out.write(floats(q));
                        }
                    }
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

    private static int readInt(DataInputStream in) throws IOException {
        return Integer.reverseBytes(in.readInt());
    }

    private static void writeInt(OutputStream out, int v) throws IOException {
        out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array());
    }

    private static byte[] floats(float[] v) {
        ByteBuffer bb = ByteBuffer.allocate(4 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        bb.asFloatBuffer().put(v);
        return bb.array();
    }

    private static float[][] readFloats(DataInputStream in, int rows, int cols) throws IOException {
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

    private static double[][] readDoubles(DataInputStream in, int rows, int cols) throws IOException {
        byte[] bytes = new byte[8 * rows * cols];
        in.readFully(bytes);
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        double[][] m = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            bb.asDoubleBuffer().get(m[r]);
            bb.position(bb.position() + 8 * cols);
        }
        return m;
    }
}
