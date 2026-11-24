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
package ai.beestream.jitcast.timesfm;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitcast.util.Json;

/**
 * A TimesFM 2.5 checkpoint's {@code config.json}, as Google publishes it on Hugging Face
 * ({@code google/timesfm-2.5-200m-pytorch}). Unknown keys fail the load, and so do sizes this engine does
 * not implement.
 *
 * @param patchLength the time steps per input patch
 * @param horizonLength the time steps one token forecasts (the output patch)
 * @param quantileHorizonLength the time steps of the continuous quantile head
 * @param hiddenSize the model width
 * @param intermediateSize the feed-forward layers' inner width
 * @param numLayers the number of transformer layers
 * @param numHeads the attention heads per layer
 * @param headDim the width of one attention head
 * @param contextLength the longest context plus horizon, in time steps
 * @param quantiles the quantile levels the model forecasts
 * @param rmsNormEps the RMSNorm epsilon
 */
public record TimesFmConfig(
        int patchLength,
        int horizonLength,
        int quantileHorizonLength,
        int hiddenSize,
        int intermediateSize,
        int numLayers,
        int numHeads,
        int headDim,
        int contextLength,
        double[] quantiles,
        float rmsNormEps) {

    private static final Set<String> KEYS = Set.of("architectures", "context_length", "head_dim", "hidden_size",
            "horizon_length", "intermediate_size", "model_type", "num_attention_heads", "num_hidden_layers",
            "patch_length", "quantile_horizon_length", "quantiles", "rms_norm_eps", "torch_compile", "torch_dtype",
            "dtype", "transformers_version", "min_timescale", "max_timescale", "use_positional_embedding",
            "attention_dropout", "initializer_range", "tolerance", "pad_val", "freq_size", "use_continuous_quantile_head");

    /** The index of the median among the model's outputs (index 0 is the mean head, then the quantiles). */
    public static final int MEDIAN_OUTPUT = 5;

    /** {@return the input patches per output patch: what one decoding step feeds back} */
    public int patchesPerOutput() {
        return horizonLength / patchLength;
    }

    /** {@return the outputs per time step: the mean head and one per quantile level} */
    public int outputs() {
        return quantiles.length + 1;
    }

    /** {@return the quantile levels, as the forecasts report them} */
    public float[] quantileLevels() {
        float[] q = new float[quantiles.length];
        for (int i = 0; i < q.length; i++) {
            q[i] = (float) quantiles[i];
        }
        return q;
    }

    /**
     * {@return the configuration in a {@code config.json} file}
     *
     * @param configJson the file
     * @throws IOException if the file cannot be read
     * @throws IllegalArgumentException if it has unknown keys or unsupported settings
     */
    public static TimesFmConfig load(Path configJson) throws IOException {
        return fromJson(Files.readString(configJson));
    }

    /**
     * {@return the configuration in a {@code config.json} document}
     *
     * @param json the document
     * @throws IllegalArgumentException if it has unknown keys or unsupported settings
     */
    public static TimesFmConfig fromJson(String json) {
        Map<String, Object> c = Json.parseObject(json);
        for (String k : c.keySet()) {
            if (!KEYS.contains(k)) {
                throw new IllegalArgumentException("unknown TimesFM config key " + k);
            }
        }
        Object type = c.get("model_type");
        if (type != null && !"timesfm".equals(type)) {
            throw new IllegalArgumentException("not a TimesFM checkpoint: model_type " + type);
        }
        String dtype = Json.str(c, "torch_dtype", Json.str(c, "dtype", "float32"));
        if (!"float32".equals(dtype)) {
            throw new IllegalArgumentException("only float32 checkpoints are supported, got " + dtype);
        }
        List<?> qs = (List<?>) c.get("quantiles");
        if (qs == null) {
            throw new IllegalArgumentException("config.json has no quantiles");
        }
        double[] q = new double[qs.size()];
        for (int i = 0; i < q.length; i++) {
            q[i] = ((Number) qs.get(i)).doubleValue();
        }
        TimesFmConfig config = new TimesFmConfig(
                Json.intVal(c, "patch_length", 32),
                Json.intVal(c, "horizon_length", 128),
                Json.intVal(c, "quantile_horizon_length", 1024),
                Json.intVal(c, "hidden_size", 1280),
                Json.intVal(c, "intermediate_size", 1280),
                Json.intVal(c, "num_hidden_layers", 20),
                Json.intVal(c, "num_attention_heads", 16),
                Json.intVal(c, "head_dim", 80),
                Json.intVal(c, "context_length", 16384),
                q,
                (float) Json.num(c, "rms_norm_eps", 1e-6));
        config.validate();
        return config;
    }

    private void validate() {
        if (patchLength < 1 || horizonLength < 1 || hiddenSize < 1 || intermediateSize < 1 || numLayers < 1 || numHeads < 1
                || headDim < 1 || contextLength < 1) {
            throw new IllegalArgumentException("non-positive model dimensions");
        }
        if (horizonLength % patchLength != 0) {
            throw new IllegalArgumentException("horizon_length " + horizonLength + " is not a multiple of patch_length " + patchLength);
        }
        if (numHeads * headDim != hiddenSize) {
            throw new IllegalArgumentException("num_attention_heads · head_dim must equal hidden_size");
        }
        if (headDim % 2 != 0) {
            throw new IllegalArgumentException("head_dim must be even for RoPE, got " + headDim);
        }
        if (quantiles.length + 1 <= MEDIAN_OUTPUT || Math.abs(quantiles[MEDIAN_OUTPUT - 1] - 0.5) > 1e-9) {
            throw new IllegalArgumentException("the quantile levels must have 0.5 at position " + MEDIAN_OUTPUT);
        }
        for (int i = 1; i < quantiles.length; i++) {
            if (!(quantiles[i] > quantiles[i - 1])) {
                throw new IllegalArgumentException("quantile levels must be strictly increasing");
            }
        }
    }
}
