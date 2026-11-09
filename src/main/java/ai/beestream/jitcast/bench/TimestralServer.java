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
import java.io.EOFException;
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
import ai.beestream.jitcast.api.ForecastEngine;
import ai.beestream.jitcast.api.TimeSeries;
import ai.beestream.jitcast.timestral.Timing;
import ai.beestream.jitcast.timestral.cpu.CpuTimestralEngine;
import ai.beestream.jitcast.timestral.tornado.TornadoTimestralEngine;

/**
 * Serves Timestral's {@code forecast()} to Python over a local socket, so benchmark harnesses
 * (jitCast-benchmarking's {@code scripts/timestral/fev_eval.py}) can score jitcast with their own data loading and metrics.
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

    /**
     * Answers one client's requests: true when it asked the server to quit, false when it disconnected
     * or sent a request whose sizes are out of range (answered with the error first).
     */
    private static boolean serve(ForecastEngine engine, DataInputStream in, OutputStream out) throws IOException {
        Wire wire = new Wire(in);
        while (true) {
            int op;
            try {
                op = wire.readInt();
            } catch (EOFException e) {
                return false;
            }
            if (op == 0) {
                return true;
            }
            wire.next();
            int n, horizon;
            List<Object[]> raw;
            try {
                n = wire.count("series count", 1, Wire.MAX_SERIES);
                horizon = wire.count("horizon", 1, 1 << 20);
                raw = new ArrayList<>(n);
                for (int i = 0; i < n; i++) {
                    int vt = wire.count("targets", 1, 1 << 16), vp = wire.count("past covariates", 0, 1 << 16);
                    int vf = wire.count("known covariates", 0, 1 << 16), cp = wire.count("categorical past covariates", 0, 1 << 16);
                    int cf = wire.count("categorical known covariates", 0, 1 << 16);
                    int t = wire.count("length", 1, Integer.MAX_VALUE), tf = wire.count("future length", 0, Integer.MAX_VALUE);
                    int tcf = wire.count("categorical future length", 0, Integer.MAX_VALUE);
                    raw.add(new Object[] {wire.floats(vt, t), vp == 0 ? null : wire.floats(vp, t), vf == 0 ? null : wire.floats(vf, tf),
                            cp == 0 ? null : wire.doubles(cp, t), cf == 0 ? null : wire.doubles(cf, tcf)});
                }
            } catch (Wire.BadRequest e) {
                Wire.error(out, e);
                out.flush();
                return false;
            }
            try {
                List<TimeSeries> series = new ArrayList<>(n);
                for (Object[] r : raw) {
                    series.add(new TimeSeries((float[][]) r[0], (float[][]) r[1], (float[][]) r[2], (double[][]) r[3], (double[][]) r[4]));
                }
                List<Forecast> result = engine.forecast(series, horizon);
                Timing.report("request of " + n + " series", 1);
                ByteArrayOutputStream body = new ByteArrayOutputStream();
                float[] levels = result.get(0).levels();
                Wire.writeInt(body, levels.length);
                Wire.writeFloats(body, levels);
                for (Forecast f : result) {
                    float[][][] v = f.values();
                    Wire.writeInt(body, v.length);
                    Wire.writeInt(body, v[0][0].length);
                    for (float[][] tq : v) {
                        for (float[] q : tq) {
                            Wire.writeFloats(body, q);
                        }
                    }
                }
                Wire.ok(out, body);
            } catch (RuntimeException e) {
                Wire.error(out, e);
                e.printStackTrace();
            }
            out.flush();
        }
    }
}
