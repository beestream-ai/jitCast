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

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * The little-endian wire format of the harness servers, read defensively: every size a client sends
 * is checked before anything is allocated, and one request may carry at most {@link #MAX_VALUES}
 * values. A request with sizes out of range fails with {@link BadRequest}; the server cannot tell
 * where it ends, so it answers with the error and drops the connection.
 */
final class Wire {

    /** The most series one request may hold. */
    static final int MAX_SERIES = 1 << 20;
    /** The most values (floats and doubles) one request may hold: 1 GiB of floats. */
    static final long MAX_VALUES = 1L << 28;

    /** A request whose sizes are out of range. */
    static final class BadRequest extends IOException {
        BadRequest(String message) {
            super(message);
        }
    }

    private final DataInputStream in;
    private long values;

    Wire(DataInputStream in) {
        this.in = in;
    }

    /** Starts a new request: its value budget is full again. */
    void next() {
        values = 0;
    }

    int readInt() throws IOException {
        return Integer.reverseBytes(in.readInt());
    }

    /** An int in {@code [min, max]}. */
    int count(String what, int min, int max) throws IOException {
        int v = readInt();
        if (v < min || v > max) {
            throw new BadRequest(what + " " + v + " is outside [" + min + ", " + max + "]");
        }
        return v;
    }

    private void reserve(int rows, int cols) throws BadRequest {
        long n = (long) rows * cols;
        if (rows < 0 || cols < 0 || values + n > MAX_VALUES) {
            throw new BadRequest("a request may hold at most " + MAX_VALUES + " values");
        }
        values += n;
    }

    float[][] floats(int rows, int cols) throws IOException {
        reserve(rows, cols);
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

    double[][] doubles(int rows, int cols) throws IOException {
        reserve(rows, cols);
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

    static void writeInt(OutputStream out, int v) throws IOException {
        out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array());
    }

    static void writeFloats(OutputStream out, float[] v) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(4 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        bb.asFloatBuffer().put(v);
        out.write(bb.array());
    }

    /** Status 0 and the response built in {@code body}. */
    static void ok(OutputStream out, ByteArrayOutputStream body) throws IOException {
        writeInt(out, 0);
        body.writeTo(out);
    }

    /** Status 1 and the message of {@code e}. */
    static void error(OutputStream out, Exception e) throws IOException {
        byte[] msg = String.valueOf(e).getBytes(StandardCharsets.UTF_8);
        writeInt(out, 1);
        writeInt(out, msg.length);
        out.write(msg);
    }
}
