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
package ai.beestream.jitcast.toto;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitcast.util.Json;

/**
 * A Toto 2.0 checkpoint's {@code config.json}, as Datadog publishes it on Hugging Face
 * ({@code Datadog/Toto-2.0-4m} to {@code Datadog/Toto-2.0-2.5B}). Unknown keys fail the load, and so do
 * settings this engine does not implement.
 *
 * @param patchSize the time steps per patch, in and out
 * @param dModel the model width
 * @param numHeads the attention heads per layer
 * @param headDim the width of one query, key and value head
 * @param dFf the gated feed-forward layers' inner width
 * @param numLayers the number of transformer layers
 * @param layerGroupSize the layers per group; the last {@code variateLayersPerGroup} of each group attend
 *        across variates, the others across time
 * @param variateLayersPerGroup the variate-attention layers per group
 * @param normEps the RMSNorm epsilon
 */
public record TotoConfig(int patchSize, int dModel, int numHeads, int headDim, int dFf, int numLayers, int layerGroupSize,
        int variateLayersPerGroup, float normEps) {

    private static final Set<String> KEYS = Set.of("attn_bias", "d_ff", "d_model", "dropout_p", "heads_per_group",
            "layer_group_size", "mlp_bias", "norm_eps", "norm_include_weight", "num_groups", "num_heads", "num_layers",
            "num_output_patches", "num_variate_layers_per_group", "patch_size", "per_dim_scale", "pre_norm", "qk_dim",
            "qk_norm", "qk_norm_include_weight", "residual_attn_ratio", "residual_mult", "use_xpos", "v_dim",
            "variate_layer_first");

    /** The quantile levels every Toto 2.0 checkpoint forecasts. */
    public static final float[] QUANTILES = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};

    /** The index of the median among {@link #QUANTILES}. */
    public static final int MEDIAN = 4;

    /** {@return the number of quantile levels} */
    public int quantiles() {
        return QUANTILES.length;
    }

    /** {@return the width of the rotated part of a query or key head: its first half} */
    public int ropeWidth() {
        return headDim / 2;
    }

    /**
     * {@return whether layer {@code i} attends across variates rather than across time}
     *
     * @param i the layer index
     */
    public boolean variateLayer(int i) {
        return i % layerGroupSize >= layerGroupSize - variateLayersPerGroup;
    }

    /**
     * {@return the configuration in a {@code config.json} file}
     *
     * @param configJson the file
     * @throws IOException if the file cannot be read
     * @throws IllegalArgumentException if it has unknown keys or unsupported settings
     */
    public static TotoConfig load(Path configJson) throws IOException {
        return fromJson(Files.readString(configJson));
    }

    /**
     * {@return the configuration in a {@code config.json} document}
     *
     * @param json the document
     * @throws IllegalArgumentException if it has unknown keys or unsupported settings
     */
    public static TotoConfig fromJson(String json) {
        Map<String, Object> c = Json.parseObject(json);
        for (String k : c.keySet()) {
            if (!KEYS.contains(k)) {
                throw new IllegalArgumentException("unknown Toto config key " + k);
            }
        }
        require(c, "attn_bias", true);
        require(c, "mlp_bias", false);
        require(c, "norm_include_weight", false);
        require(c, "qk_norm", false);
        require(c, "per_dim_scale", true);
        require(c, "use_xpos", true);
        require(c, "pre_norm", true);
        require(c, "variate_layer_first", false);
        if (Json.num(c, "dropout_p", 0) != 0) {
            throw new IllegalArgumentException("dropout_p must be 0");
        }
        if (Json.intVal(c, "num_output_patches", 1) != 1) {
            throw new IllegalArgumentException("only num_output_patches 1 is supported");
        }
        int heads = Json.intVal(c, "num_heads", 0);
        int d = Json.intVal(c, "d_model", 0);
        if (Json.intVal(c, "num_groups", heads) != heads) {
            throw new IllegalArgumentException("grouped-query attention is not supported");
        }
        int qk = Json.intVal(c, "qk_dim", heads == 0 ? 0 : d / heads);
        if (Json.intVal(c, "v_dim", qk) != qk) {
            throw new IllegalArgumentException("v_dim must equal qk_dim");
        }
        int dff = Json.intVal(c, "d_ff", (int) (4 * d * 2 / 3 + 7) / 8 * 8);
        TotoConfig config = new TotoConfig(Json.intVal(c, "patch_size", 32), d, heads, qk, dff, Json.intVal(c, "num_layers", 0),
                Json.intVal(c, "layer_group_size", 0), Json.intVal(c, "num_variate_layers_per_group", 0),
                (float) Json.num(c, "norm_eps", 5e-5));
        config.validate();
        return config;
    }

    private static void require(Map<String, Object> c, String key, boolean value) {
        Object v = c.get(key);
        if (v != null && !Boolean.valueOf(value).equals(v)) {
            throw new IllegalArgumentException(key + " must be " + value + ", got " + v);
        }
    }

    private void validate() {
        if (patchSize < 1 || dModel < 1 || numHeads < 1 || headDim < 1 || dFf < 1 || numLayers < 1 || layerGroupSize < 1) {
            throw new IllegalArgumentException("non-positive model dimensions");
        }
        if (numHeads * headDim != dModel) {
            throw new IllegalArgumentException("num_heads · qk_dim must equal d_model");
        }
        if (headDim % 4 != 0) {
            throw new IllegalArgumentException("qk_dim must be a multiple of 4 for the half-width rotary embedding");
        }
        if (numLayers % layerGroupSize != 0 || variateLayersPerGroup < 0 || variateLayersPerGroup > layerGroupSize) {
            throw new IllegalArgumentException("inconsistent layer groups");
        }
    }
}
