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
package ai.beestream.jitcast.t0;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitcast.util.Json;

/**
 * A t0 checkpoint's {@code config.json}, as The Forecasting Company publishes it on Hugging Face
 * ({@code theforecastingcompany/t0-alpha}, {@code t0-beta}, {@code t0-gamma}). Fields a checkpoint leaves
 * out take the reference's defaults (those of t0-alpha). Unknown keys fail the load, and so do settings
 * this engine does not implement.
 *
 * @param embedDim the model width
 * @param numLayers the number of transformer layers
 * @param numHeads the attention heads per layer
 * @param mlpHiddenDim the inner width of each layer's SwiGLU feed-forward
 * @param patchSize the time steps per patch
 * @param groupEveryN every {@code groupEveryN}-th layer attends across the variates of a series instead of
 *     across time; 0 or below for none
 * @param quantileLevels the quantile levels the model forecasts, ascending
 * @param scalerUseArcsinh whether the scaler applies {@code asinh} after standardizing
 * @param scalerEps the floor that keeps a standard deviation away from zero
 * @param scalerEpsVarianceOffset {@code true} when {@code scalerEps} is added to the variance before the square
 *     root ({@code variance_offset}), {@code false} when it bounds the deviation from below ({@code std_clamp})
 * @param xposHalves {@code true} when the xPos scales repeat over the two halves of a head ({@code halves}),
 *     {@code false} when adjacent dimensions share one ({@code interleaved})
 * @param scalerZeroNoValue whether cells without an observation enter the model as zero after scaling
 */
public record T0Config(
        int embedDim,
        int numLayers,
        int numHeads,
        int mlpHiddenDim,
        int patchSize,
        int groupEveryN,
        double[] quantileLevels,
        boolean scalerUseArcsinh,
        float scalerEps,
        boolean scalerEpsVarianceOffset,
        boolean xposHalves,
        boolean scalerZeroNoValue) {

    private static final Set<String> KEYS = Set.of("dropout", "dtype", "embed_dim", "group_every_n", "mlp_hidden_dim",
            "num_heads", "num_layers", "patch_size", "quantile_levels", "scaler_eps", "scaler_eps_mode",
            "scaler_use_arcsinh", "scaler_zero_no_value", "xpos_scale_layout");

    /** The RMSNorm epsilon of every norm in the model. */
    public static final float NORM_EPS = 1e-8f;
    /** The rotary embedding's base frequency. */
    public static final double ROPE_THETA = 10_000;
    /** The xPos scale base: the distance over which a position's scale changes by its per-dimension factor. */
    public static final float XPOS_SCALE_BASE = 512;
    /** The longest horizon one forward pass predicts; longer horizons continue with the rollout. */
    public static final int MAX_HORIZON = 1024;

    /** {@return the width of one attention head} */
    public int headDim() {
        return embedDim / numHeads;
    }

    /** {@return the number of trained quantile levels} */
    public int quantiles() {
        return quantileLevels.length;
    }

    /**
     * {@return whether layer {@code i} attends across the variates of a series rather than across time}
     *
     * @param i the layer index, from 0
     */
    public boolean isGroupLayer(int i) {
        return groupEveryN > 0 && (i + 1) % groupEveryN == 0;
    }

    /** {@return the trained quantile levels, as the forecasts report them} */
    public float[] levels() {
        float[] q = new float[quantileLevels.length];
        for (int i = 0; i < q.length; i++) {
            q[i] = (float) quantileLevels[i];
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
    public static T0Config load(Path configJson) throws IOException {
        return fromJson(Files.readString(configJson));
    }

    /**
     * {@return the configuration in a {@code config.json} document}
     *
     * @param json the document
     * @throws IllegalArgumentException if it has unknown keys or unsupported settings
     */
    public static T0Config fromJson(String json) {
        Map<String, Object> c = Json.parseObject(json);
        for (String k : c.keySet()) {
            if (!KEYS.contains(k)) {
                throw new IllegalArgumentException("unknown t0 config key " + k);
            }
        }
        Object dtype = c.get("dtype");
        if (dtype != null && !"float32".equals(dtype)) {
            throw new IllegalArgumentException("only float32 inference is supported, got dtype " + dtype);
        }
        List<?> qs = (List<?>) c.get("quantile_levels");
        if (qs == null || qs.isEmpty()) {
            throw new IllegalArgumentException("config.json has no quantile_levels");
        }
        double[] q = new double[qs.size()];
        for (int i = 0; i < q.length; i++) {
            q[i] = ((Number) qs.get(i)).doubleValue();
        }
        String epsMode = Json.str(c, "scaler_eps_mode", "variance_offset");
        if (!epsMode.equals("variance_offset") && !epsMode.equals("std_clamp")) {
            throw new IllegalArgumentException("unknown scaler_eps_mode " + epsMode);
        }
        String layout = Json.str(c, "xpos_scale_layout", "interleaved");
        if (!layout.equals("interleaved") && !layout.equals("halves")) {
            throw new IllegalArgumentException("unknown xpos_scale_layout " + layout);
        }
        T0Config config = new T0Config(
                required(c, "embed_dim"),
                required(c, "num_layers"),
                required(c, "num_heads"),
                required(c, "mlp_hidden_dim"),
                required(c, "patch_size"),
                required(c, "group_every_n"),
                q,
                Json.bool(c, "scaler_use_arcsinh", true),
                (float) Json.num(c, "scaler_eps", 0.1),
                epsMode.equals("variance_offset"),
                layout.equals("halves"),
                Json.bool(c, "scaler_zero_no_value", true));
        config.validate();
        return config;
    }

    private static int required(Map<String, Object> c, String key) {
        if (!(c.get(key) instanceof Number n)) {
            throw new IllegalArgumentException("config.json has no " + key);
        }
        return n.intValue();
    }

    private void validate() {
        if (embedDim < 1 || numLayers < 1 || numHeads < 1 || mlpHiddenDim < 1 || patchSize < 1) {
            throw new IllegalArgumentException("non-positive model dimensions");
        }
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embed_dim must be divisible by num_heads");
        }
        if (headDim() % 2 != 0) {
            throw new IllegalArgumentException("the head width must be even for the rotary embedding, got " + headDim());
        }
        if (groupEveryN > 0 && numLayers % groupEveryN != 0) {
            throw new IllegalArgumentException("group_every_n must divide num_layers");
        }
        if (MAX_HORIZON % patchSize != 0) {
            throw new IllegalArgumentException("patch_size must divide " + MAX_HORIZON);
        }
        if (!(scalerEps > 0)) {
            throw new IllegalArgumentException("scaler_eps must be positive");
        }
        for (int i = 0; i < quantileLevels.length; i++) {
            if (!(quantileLevels[i] > 0 && quantileLevels[i] < 1) || i > 0 && !(quantileLevels[i] > quantileLevels[i - 1])) {
                throw new IllegalArgumentException("quantile_levels must be strictly increasing in (0, 1)");
            }
        }
    }
}
