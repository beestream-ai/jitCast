package ai.beestream.jitllm.tirex.io;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;

public class SafeTensorsTest {

    private static Path write(String header, float[] data) throws IOException {
        byte[] h = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buf = ByteBuffer.allocate(8 + h.length + 4 * data.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(h.length).put(h);
        for (float f : data) {
            buf.putFloat(f);
        }
        Path file = Files.createTempFile("st", ".safetensors");
        Files.write(file, buf.array());
        file.toFile().deleteOnExit();
        return file;
    }

    @Test
    public void readsTensorsAndMetadata() throws IOException {
        // a 3-byte odd header length, so the data section starts unaligned
        String header = "{\"__metadata__\":{\"k\":\"v\"},"
                + "\"a\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
                + "\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[24,32]}}   ";
        Path file = write(header, new float[] {1, 2, 3, 4, 5, 6, -1.5f, 7.25f});
        try (SafeTensors st = SafeTensors.open(file)) {
            assertEquals("v", st.metadata().get("k"));
            assertEquals(2, st.tensors().size());
            Tensor a = st.get("a").requireF32(2, 3);
            assertEquals(3, a.dim(-1));
            assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6}, a.toFloatArray(), 0f);
            assertEquals(7.25f, st.get("b").getFloat(1), 0f);
        }
    }

    @Test
    public void rejectsShapeAndRangeMismatch() throws IOException {
        Path file = write("{\"a\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,8]}}", new float[] {1, 2});
        assertThrows(IllegalArgumentException.class, () -> SafeTensors.open(file));
        Path ok = write("{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", new float[] {1, 2});
        try (SafeTensors st = SafeTensors.open(ok)) {
            assertThrows(IllegalArgumentException.class, () -> st.get("a").requireF32(1, 2));
        }
    }
}
