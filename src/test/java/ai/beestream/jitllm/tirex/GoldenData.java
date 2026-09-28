package ai.beestream.jitllm.tirex;

import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.beehive.jitllm.server.Json;
import ai.beestream.jitllm.tirex.io.SafeTensors;

/**
 * Locates the converted weights ({@code -Dtirex.weights}) and a golden directory
 * ({@code -Dtirex.golden}) written by {@code scripts/dump_golden.py}. Tests that need them are
 * skipped, not failed, when they are not configured.
 */
public final class GoldenData {

    private GoldenData() {}

    public static Path weights() {
        return required("tirex.weights");
    }

    public static Path goldenDir() {
        return required("tirex.golden");
    }

    /** Case names from {@code manifest.json}, in dump order. */
    @SuppressWarnings("unchecked")
    public static List<String> cases() throws IOException {
        Map<String, Object> manifest = Json.parseObject(Files.readString(goldenDir().resolve("manifest.json")));
        return List.copyOf(((Map<String, Object>) manifest.get("cases")).keySet());
    }

    public static SafeTensors open(String testCase) throws IOException {
        return SafeTensors.open(goldenDir().resolve(testCase + ".safetensors"));
    }

    private static Path required(String property) {
        String value = System.getProperty(property, "");
        assumeTrue("-D" + property + " not set", !value.isBlank());
        Path path = Path.of(value);
        assumeTrue(path + " does not exist", Files.exists(path));
        return path;
    }

    /** max|a-b| / max(max|b|, 1e-6): one number that is comparable across stages. */
    public static double relativeMaxError(float[] actual, float[] expected) {
        if (actual.length != expected.length) {
            throw new AssertionError("length " + actual.length + " != " + expected.length);
        }
        double diff = 0, scale = 1e-6;
        for (int i = 0; i < actual.length; i++) {
            diff = Math.max(diff, Math.abs(actual[i] - expected[i]));
            scale = Math.max(scale, Math.abs(expected[i]));
        }
        return diff / scale;
    }
}
