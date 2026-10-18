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
package ai.beestream.jitrex.timestral;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitrex.util.Json;

/**
 * A Timestral checkpoint's {@code config.json}: the fields that change the forward pass and the
 * decoding, as {@code timestral.config.TimestralConfig}. Unknown keys fail the load, as in the
 * reference, and so do the settings this engine does not implement.
 *
 * @param layers one entry per transformer layer: its attention axes in order, {@code true} for
 *     time and {@code false} for variate, each followed by the layer's one MLP
 */
public record TimestralConfig(
        int patchSize,
        int dModel,
        int numHeads,
        int mlpHiddenDim,
        List<boolean[]> layers,
        double[] quantileKnots,
        int blockSize,
        int contextLength,
        float ropeTheta,
        float ropePartialFactor,
        float normEps,
        boolean timeAttnGate,
        boolean variateAttnGate,
        float minimumScale,
        boolean useAsinh,
        boolean covariateEnabled,
        boolean attendMissingPatches,
        float quantileWidthFactor) {

    private static final Set<String> KEYS = Set.of("patch_size", "d_model", "num_heads", "mlp_hidden_dim",
            "block_schedule", "quantile_knots", "block_size", "context_length", "rope_theta", "rope_partial_factor",
            "norm_eps", "attn_bias", "mlp_bias", "time_attn_gate", "variate_attn_gate", "minimum_scale", "use_asinh",
            "stabilize_with_global", "covariate_enabled", "split_context_horizon_attn", "attend_missing_patches",
            "quantile_width_factor");

    public int headDim() {
        return dModel / numHeads;
    }

    /** Head dimensions rotated by RoPE; the rest pass through unrotated. */
    public int ropeDim() {
        return (int) (headDim() * ropePartialFactor);
    }

    /** The quantile levels, as the forecasts report them. */
    public float[] quantileLevels() {
        float[] q = new float[quantileKnots.length];
        for (int i = 0; i < q.length; i++) {
            q[i] = (float) quantileKnots[i];
        }
        return q;
    }

    public int numQuantiles() {
        return quantileKnots.length;
    }

    public int medianIndex() {
        for (int i = 0; i < quantileKnots.length; i++) {
            if (quantileKnots[i] == 0.5) {
                return i;
            }
        }
        throw new IllegalStateException("no 0.5 quantile knot");
    }

    /** The patches of the trained context length. */
    public int contextPatches() {
        return contextLength / patchSize;
    }

    /**
     * Whether the layer schedule can route a known covariate's future to the targets: some time
     * attention must be followed by at least one more time attention and at least one variate one.
     */
    public boolean supportsKnownCovariates() {
        if (!covariateEnabled) {
            return false;
        }
        List<Boolean> ops = new ArrayList<>();
        for (boolean[] axes : layers) {
            for (boolean isTime : axes) {
                ops.add(isTime);
            }
        }
        for (int i = 0; i < ops.size(); i++) {
            List<Boolean> rest = ops.subList(i + 1, ops.size());
            if (ops.get(i) && rest.contains(true) && rest.contains(false)) {
                return true;
            }
        }
        return false;
    }

    public static TimestralConfig load(Path configJson) throws IOException {
        return fromJson(Files.readString(configJson));
    }

    public static TimestralConfig fromJson(String json) {
        Map<String, Object> c = Json.parseObject(json);
        for (String key : c.keySet()) {
            if (!KEYS.contains(key)) {
                throw new IllegalArgumentException("unknown config key: " + key);
            }
        }
        requireFalse(c, "attn_bias", false);
        requireFalse(c, "mlp_bias", false);
        requireFalse(c, "stabilize_with_global", false);

        List<boolean[]> layers = new ArrayList<>();
        for (Object kind : (List<?>) c.get("block_schedule")) {
            layers.add(switch ((String) kind) {
                case "time" -> new boolean[] {true};
                case "variate" -> new boolean[] {false};
                case "time_variate" -> new boolean[] {true, false};
                case "variate_time" -> new boolean[] {false, true};
                default -> throw new IllegalArgumentException("unknown layer kind " + kind);
            });
        }
        List<?> knots = (List<?>) c.get("quantile_knots");
        double[] q = new double[knots.size()];
        for (int i = 0; i < q.length; i++) {
            q[i] = ((Number) knots.get(i)).doubleValue();
        }
        TimestralConfig config = new TimestralConfig(
                Json.intVal(c, "patch_size", 0),
                Json.intVal(c, "d_model", 0),
                Json.intVal(c, "num_heads", 0),
                Json.intVal(c, "mlp_hidden_dim", 0),
                List.copyOf(layers),
                q,
                Json.intVal(c, "block_size", 0),
                Json.intVal(c, "context_length", 0),
                (float) Json.num(c, "rope_theta", 10000.0),
                (float) Json.num(c, "rope_partial_factor", 0.5),
                (float) Json.num(c, "norm_eps", 1e-6),
                gate(c, "time_attn_gate"),
                gate(c, "variate_attn_gate"),
                (float) Json.num(c, "minimum_scale", 1e-6),
                Json.bool(c, "use_asinh", true),
                Json.bool(c, "covariate_enabled", true),
                Json.bool(c, "attend_missing_patches", true),
                (float) Json.num(c, "quantile_width_factor", 2.0));
        config.validate();
        return config;
    }

    private void validate() {
        if (patchSize < 1 || dModel < 1 || numHeads < 1 || dModel % numHeads != 0) {
            throw new IllegalArgumentException("d_model " + dModel + " is not divisible by num_heads " + numHeads);
        }
        if (ropeDim() <= 0 || ropeDim() % 2 != 0) {
            throw new IllegalArgumentException("rope_partial_factor " + ropePartialFactor + " gives rope_dim " + ropeDim());
        }
        for (int i = 1; i < quantileKnots.length; i++) {
            if (!(quantileKnots[i - 1] < quantileKnots[i])) {
                throw new IllegalArgumentException("quantile_knots must be increasing");
            }
        }
        medianIndex();
        if (contextLength % patchSize != 0) {
            throw new IllegalArgumentException("context_length must be a multiple of patch_size");
        }
    }

    private static boolean gate(Map<String, Object> c, String key) {
        String g = Json.str(c, key, "sdpa_sigmoid");
        return switch (g) {
            case "sdpa_sigmoid" -> true;
            case "none" -> false;
            default -> throw new IllegalArgumentException("unknown attention gate " + g);
        };
    }

    /** Fails on a setting this engine does not implement. */
    private static void requireFalse(Map<String, Object> c, String key, boolean def) {
        if (Json.bool(c, key, def)) {
            throw new IllegalArgumentException(key + " = true is not supported");
        }
    }
}
