package ai.beestream.jitrex.io;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitrex.util.Json;

/**
 * A read-only, memory-mapped <a href="https://huggingface.co/docs/safetensors">safetensors</a>
 * file: an 8-byte little-endian header length, a JSON header naming each tensor's dtype, shape and
 * byte range, then the tensor bytes. Tensors are slices of one mapping, so opening a file copies
 * nothing; the mapping lives until {@link #close()}.
 */
public final class SafeTensors implements AutoCloseable {

    /** Little-endian float, the only element layout safetensors writes for F32. */
    public static final ValueLayout.OfFloat F32 =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final Arena arena;
    private final Map<String, Tensor> tensors;
    private final Map<String, String> metadata;

    private SafeTensors(Arena arena, Map<String, Tensor> tensors, Map<String, String> metadata) {
        this.arena = arena;
        this.tensors = tensors;
        this.metadata = metadata;
    }

    /** Maps {@code path} and parses its header. */
    public static SafeTensors open(Path path) throws IOException {
        Arena arena = Arena.ofShared();
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            MemorySegment file = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), arena);
            long headerLength = file.get(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 0);
            if (headerLength <= 0 || 8 + headerLength > file.byteSize()) {
                throw new IOException(path + ": bad safetensors header length " + headerLength);
            }
            byte[] headerBytes = file.asSlice(8, headerLength).toArray(ValueLayout.JAVA_BYTE);
            Map<String, Object> header = Json.parseObject(new String(headerBytes, StandardCharsets.UTF_8));
            MemorySegment data = file.asSlice(8 + headerLength);

            Map<String, Tensor> tensors = new LinkedHashMap<>();
            Map<String, String> metadata = Map.of();
            for (Map.Entry<String, Object> entry : header.entrySet()) {
                @SuppressWarnings("unchecked")
                Map<String, Object> info = (Map<String, Object>) entry.getValue();
                if (entry.getKey().equals("__metadata__")) {
                    Map<String, String> meta = new LinkedHashMap<>();
                    info.forEach((k, v) -> meta.put(k, String.valueOf(v)));
                    metadata = Collections.unmodifiableMap(meta);
                    continue;
                }
                tensors.put(entry.getKey(), tensor(entry.getKey(), info, data));
            }
            return new SafeTensors(arena, Collections.unmodifiableMap(tensors), metadata);
        } catch (IOException | RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    private static Tensor tensor(String name, Map<String, Object> info, MemorySegment data) {
        DType dtype = DType.valueOf((String) info.get("dtype"));
        List<?> shapeList = (List<?>) info.get("shape");
        long[] shape = new long[shapeList.size()];
        long elements = 1;
        for (int i = 0; i < shape.length; i++) {
            shape[i] = ((Number) shapeList.get(i)).longValue();
            elements *= shape[i];
        }
        List<?> offsets = (List<?>) info.get("data_offsets");
        long begin = ((Number) offsets.get(0)).longValue();
        long end = ((Number) offsets.get(1)).longValue();
        if (end - begin != elements * dtype.bytes) {
            throw new IllegalArgumentException(name + ": byte range " + (end - begin)
                    + " does not match shape " + java.util.Arrays.toString(shape) + " of " + dtype);
        }
        return new Tensor(name, dtype, shape, data.asSlice(begin, end - begin));
    }

    public Map<String, Tensor> tensors() {
        return tensors;
    }

    public Tensor get(String name) {
        Tensor t = tensors.get(name);
        if (t == null) {
            throw new IllegalArgumentException("no tensor '" + name + "'");
        }
        return t;
    }

    public boolean contains(String name) {
        return tensors.containsKey(name);
    }

    /** The {@code __metadata__} string map; empty if the file has none. */
    public Map<String, String> metadata() {
        return metadata;
    }

    @Override
    public void close() {
        arena.close();
    }

    /** Element types this reader understands. */
    public enum DType {
        F32(4), F16(2), BF16(2), F64(8), I64(8), I32(4), U8(1), BOOL(1);

        public final int bytes;

        DType(int bytes) {
            this.bytes = bytes;
        }
    }
}
