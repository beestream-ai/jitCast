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
package ai.beestream.jitcast.chronos2;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitcast.util.Json;

/**
 * A Chronos-2 checkpoint's {@code config.json}: the transformer's {@code Chronos2CoreConfig} and the
 * forecasting {@code chronos_config}, as Amazon publishes them on Hugging Face. Unknown keys fail
 * the load, and so do the settings this engine does not implement (gated or non-ReLU feed-forward
 * layers, overlapping patches, different input and output patch sizes).
 *
 * @param timeEncodingScale the divisor of the time index fed to the patch embedding
 * @param maxOutputPatches the patches one forward pass forecasts before the long-horizon unrolling
 */
public record Chronos2Config(
        int dModel,
        int dKv,
        int dFf,
        int numLayers,
        int numHeads,
        float layerNormEps,
        float ropeTheta,
        int contextLength,
        int patchSize,
        int maxOutputPatches,
        double[] quantiles,
        boolean useRegToken,
        boolean useArcsinh,
        int timeEncodingScale) {

    private static final Set<String> KEYS = Set.of("architectures", "chronos_config", "chronos_pipeline_class", "d_ff", "d_kv",
            "d_model", "dense_act_fn", "dropout_rate", "feed_forward_proj", "initializer_factor", "is_gated_act",
            "layer_norm_epsilon", "model_type", "num_heads", "num_layers", "pad_token_id", "reg_token_id", "rope_theta",
            "torch_dtype", "dtype", "transformers_version", "use_cache", "vocab_size", "attn_implementation");
    private static final Set<String> CHRONOS_KEYS = Set.of("context_length", "input_patch_size", "input_patch_stride",
            "output_patch_size", "max_output_patches", "quantiles", "use_reg_token", "use_arcsinh", "time_encoding_scale");

    /** {@code num_heads · d_kv}, the width of q, k and v. */
    public int inner() {
        return numHeads * dKv;
    }

    public int numQuantiles() {
        return quantiles.length;
    }

    /** The quantile levels, as the forecasts report them. */
    public float[] quantileLevels() {
        float[] q = new float[quantiles.length];
        for (int i = 0; i < q.length; i++) {
            q[i] = (float) quantiles[i];
        }
        return q;
    }

    /** The rows of the learned token table: the padding token, and {@code [REG]} when used. */
    public int vocabSize() {
        return useRegToken ? 2 : 1;
    }

    /** The steps one forward pass forecasts before the long-horizon unrolling. */
    public int predictionLength() {
        return maxOutputPatches * patchSize;
    }

    public static Chronos2Config load(Path configJson) throws IOException {
        return fromJson(Files.readString(configJson));
    }

    @SuppressWarnings("unchecked")
    public static Chronos2Config fromJson(String json) {
        Map<String, Object> c = Json.parseObject(json);
        requireKnown(c, KEYS);
        Map<String, Object> cc = (Map<String, Object>) c.get("chronos_config");
        if (cc == null) {
            throw new IllegalArgumentException("not a Chronos-2 config: no chronos_config");
        }
        requireKnown(cc, CHRONOS_KEYS);
        Object arch = c.get("architectures");
        if (arch instanceof List<?> l && !l.contains("Chronos2Model")) {
            throw new IllegalArgumentException("not a Chronos-2 checkpoint: " + l);
        }
        String act = Json.str(c, "dense_act_fn", Json.str(c, "feed_forward_proj", "relu"));
        if (!"relu".equals(act) || Json.bool(c, "is_gated_act", false)
                || !"relu".equals(Json.str(c, "feed_forward_proj", "relu"))) {
            throw new IllegalArgumentException("only the ReLU feed-forward layer is supported, got " + act);
        }
        String dtype = Json.str(c, "torch_dtype", Json.str(c, "dtype", "float32"));
        if (!"float32".equals(dtype)) {
            throw new IllegalArgumentException("only float32 checkpoints are supported, got " + dtype);
        }
        int in = Json.intVal(cc, "input_patch_size", 0);
        if (in != Json.intVal(cc, "output_patch_size", 0) || in != Json.intVal(cc, "input_patch_stride", 0)) {
            throw new IllegalArgumentException("input and output patch sizes and the patch stride must be equal");
        }
        List<?> qs = (List<?>) cc.get("quantiles");
        double[] q = new double[qs.size()];
        for (int i = 0; i < q.length; i++) {
            q[i] = ((Number) qs.get(i)).doubleValue();
        }
        int context = Json.intVal(cc, "context_length", 0);
        Chronos2Config config = new Chronos2Config(
                Json.intVal(c, "d_model", 512),
                Json.intVal(c, "d_kv", 64),
                Json.intVal(c, "d_ff", 2048),
                Json.intVal(c, "num_layers", 6),
                Json.intVal(c, "num_heads", 8),
                (float) Json.num(c, "layer_norm_epsilon", 1e-6),
                (float) Json.num(c, "rope_theta", 10000.0),
                context,
                in,
                Json.intVal(cc, "max_output_patches", 1),
                q,
                Json.bool(cc, "use_reg_token", false),
                Json.bool(cc, "use_arcsinh", false),
                Json.intVal(cc, "time_encoding_scale", context));
        config.validate();
        return config;
    }

    private void validate() {
        if (dModel < 1 || dKv < 1 || dFf < 1 || numLayers < 1 || numHeads < 1 || patchSize < 1 || maxOutputPatches < 1) {
            throw new IllegalArgumentException("non-positive model dimensions");
        }
        if (dKv % 2 != 0) {
            throw new IllegalArgumentException("d_kv must be even for RoPE, got " + dKv);
        }
        if (contextLength < patchSize || timeEncodingScale < 1) {
            throw new IllegalArgumentException("context_length must hold a patch");
        }
        for (int i = 0; i < quantiles.length; i++) {
            if (!(quantiles[i] > 0 && quantiles[i] < 1) || i > 0 && !(quantiles[i - 1] < quantiles[i])) {
                throw new IllegalArgumentException("quantiles must be increasing and in (0, 1)");
            }
        }
    }

    private static void requireKnown(Map<String, Object> m, Set<String> keys) {
        for (String key : m.keySet()) {
            if (!keys.contains(key)) {
                throw new IllegalArgumentException("unknown config key: " + key);
            }
        }
    }
}
