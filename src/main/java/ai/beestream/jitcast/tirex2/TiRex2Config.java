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
package ai.beestream.jitcast.tirex2;

import java.util.List;
import java.util.Map;

import ai.beestream.jitcast.util.Json;

/**
 * The TiRex-2 hyper-parameters, read from the {@code model-config.yaml} that
 * {@code scripts/convert_ckpt.py} embeds as JSON under {@code tirex2.config}.
 *
 * <p>Only the values the forward pass needs are kept. Constants that the reference code hard-codes
 * (not in the config) are here as well, so every number the engine uses has one home.
 *
 * @param numBlocks the number of blocks
 * @param embeddingDim the model width
 * @param patchSize the time steps per patch
 * @param inputFfDim the inner width of the input patch embedding
 * @param outputFfDim the inner width of the output patch embedding
 * @param contextLen the model context in time steps
 * @param futureLen the longest horizon in time steps
 * @param quantiles the quantile levels the model forecasts, an odd count with 0.5 in the middle
 * @param useArcsinh whether the scaler applies arcsinh
 * @param binaryAware whether binary series are scaled as such
 * @param stackOutNormEps the epsilon of the norm after the last block
 * @param ttaDiff whether trend differencing is on by default
 * @param ttaSignFlip whether sign-flip test-time augmentation is on by default
 * @param recipe the type of each block, in order
 * @param timeMixerHeads the mLSTM heads
 * @param slstmHeads the sLSTM heads
 * @param variateHeads the variate attention heads
 * @param useQkNorm whether variate attention normalizes q and k
 */
public record TiRex2Config(
        int numBlocks,
        int embeddingDim,
        int patchSize,
        int inputFfDim,
        int outputFfDim,
        int contextLen,
        int futureLen,
        float[] quantiles,
        boolean useArcsinh,
        boolean binaryAware,
        float stackOutNormEps,
        boolean ttaDiff,
        boolean ttaSignFlip,
        List<BlockType> recipe,
        int timeMixerHeads,
        int slstmHeads,
        int variateHeads,
        boolean useQkNorm) {

    /**
     * Checks the sizes and quantile levels.
     *
     * @throws IllegalArgumentException if the patches, heads or levels are inconsistent
     */
    public TiRex2Config {
        if (patchSize < 1 || contextLen < 1 || futureLen < 0 || (contextLen + futureLen) % patchSize != 0) {
            throw new IllegalArgumentException("context_len + future_len (" + contextLen + " + " + futureLen
                    + ") must be a positive multiple of the patch size " + patchSize);
        }
        for (int h : new int[] {timeMixerHeads, slstmHeads, variateHeads}) {
            if (h < 1 || embeddingDim % h != 0) {
                throw new IllegalArgumentException("embedding_dim " + embeddingDim + " is not divisible by " + h + " heads");
            }
        }
        // the median is the middle level: calibration and the point forecast take it from there
        if (quantiles.length % 2 == 0 || quantiles[quantiles.length / 2] != 0.5f) {
            throw new IllegalArgumentException("the quantiles must be an odd number of levels with 0.5 in the middle");
        }
        for (int i = 0; i < quantiles.length; i++) {
            if (!(quantiles[i] > 0f && quantiles[i] < 1f) || i > 0 && quantiles[i] <= quantiles[i - 1]) {
                throw new IllegalArgumentException("the quantiles must increase strictly within (0, 1)");
            }
        }
    }

    /** The time mixer of a block. Every block's variate mixer is attention. */
    public enum BlockType {
        /** A matrix-memory LSTM time mixer. */
        MLSTM,
        /** A scalar-memory LSTM time mixer. */
        SLSTM
    }

    // Constants of the reference implementation (xlstm-large and tirex2 defaults).
    public static final float NORM_EPS = 1e-6f; // RMSNorm, per-head LayerNorms, attention norms
    public static final float MLSTM_EPS = 1e-6f;
    public static final float GATE_SOFT_CAP = 15f;
    public static final float QK_DIM_FACTOR = 0.5f;
    public static final int FFN_ROUND_TO = 64;
    public static final float FFN_PROJ_FACTOR = 2.6667f;
    public static final int ATTENTION_FFN_FACTOR = 4;
    public static final float SCALER_EPS = 1e-8f;
    public static final float SINH_CLAMP = 20f;

    /** {@return the patches of context plus horizon} */
    public int numPatches() {
        return (contextLen + futureLen) / patchSize;
    }

    /** {@return the time steps of context plus horizon} */
    public int paddedLength() {
        return contextLen + futureLen;
    }

    /** {@return the number of quantile levels} */
    public int numQuantiles() {
        return quantiles.length;
    }

    /** {@return the width of the mLSTM's queries and keys} */
    public int mlstmQkDim() {
        return (int) (embeddingDim * QK_DIM_FACTOR);
    }

    /** {@return the inner width of the feed-forward layers} */
    public int ffnDim() {
        int raw = (int) (embeddingDim * FFN_PROJ_FACTOR);
        return (raw + FFN_ROUND_TO - 1) / FFN_ROUND_TO * FFN_ROUND_TO;
    }

    /**
     * {@return the configuration embedded in a converted checkpoint}
     *
     * @param json the {@code tirex2.config} metadata of the safetensors file
     * @throws IllegalArgumentException if a key is missing or a value is invalid
     */
    public static TiRex2Config fromJson(String json) {
        Map<String, Object> c = Json.parseObject(json);
        int d = intOf(c, "embedding_dim");
        int in = intOf(c, "input_patch_size");
        int out = intOf(c, "output_patch_size");
        Map<String, Object> tok = map(c, "tokenizer_cfg");
        if (in != out || intOf(tok, "input_patch_stride") != in) {
            throw new IllegalArgumentException("only non-overlapping patches with input == output size are supported");
        }
        Map<String, Object> scaler = map(c, "scaler_cfg");
        Map<String, Object> outNorm = optionalMap(c, "stack_out_norm_config");
        if (outNorm != null && Boolean.TRUE.equals(outNorm.get("use_rmsnorm"))) {
            throw new IllegalArgumentException("stack_out_norm use_rmsnorm=true is not supported");
        }

        Map<String, Object> stack = map(c, "stack_config");
        Map<String, Object> templates = map(stack, "templates");
        List<?> recipeNames = (List<?>) stack.get("recipe");
        List<BlockType> recipe = recipeNames.stream().map(n -> {
            Map<String, Object> mixer = map(map(templates, (String) n), "time_mixer");
            return switch ((String) mixer.get("model_type")) {
                case "bi-mlstm" -> BlockType.MLSTM;
                case "bi-slstm" -> BlockType.SLSTM;
                default -> throw new IllegalArgumentException("unknown time mixer " + mixer.get("model_type"));
            };
        }).toList();
        int numBlocks = intOf(c, "num_blocks");
        if (recipe.size() != numBlocks) {
            throw new IllegalArgumentException("recipe has " + recipe.size() + " entries for " + numBlocks + " blocks");
        }

        int heads = -1, slstmHeads = -1, variateHeads = -1;
        for (Object t : templates.values()) {
            @SuppressWarnings("unchecked")
            Map<String, Object> template = (Map<String, Object>) t;
            Map<String, Object> tm = map(template, "time_mixer");
            Map<String, Object> vm = map(template, "variate_mixer");
            heads = same(heads, intOf(tm, "num_heads"), "time mixer num_heads");
            if (tm.get("num_slstm_heads") != null) {
                slstmHeads = same(slstmHeads, intOf(tm, "num_slstm_heads"), "num_slstm_heads");
            }
            variateHeads = same(variateHeads, intOf(vm, "num_heads"), "variate mixer num_heads");
            if (intOf(tm, "embedding_dim") != d || intOf(vm, "embedding_dim") != d) {
                throw new IllegalArgumentException("mixer embedding_dim differs from the model's");
            }
        }

        List<?> q = (List<?>) c.get("quantiles");
        float[] quantiles = new float[q.size()];
        for (int i = 0; i < quantiles.length; i++) {
            quantiles[i] = ((Number) q.get(i)).floatValue();
        }
        return new TiRex2Config(
                numBlocks, d, in,
                intOf(c, "input_ff_dim"),
                (int) (d * ((Number) c.get("h_expand")).doubleValue()),
                intOf(c, "context_len"),
                intOf(c, "future_len"),
                quantiles,
                Boolean.TRUE.equals(scaler.get("use_arcsinh")),
                Boolean.TRUE.equals(scaler.get("binaryaware")),
                outNorm == null ? Float.NaN : ((Number) outNorm.getOrDefault("eps", 1e-7)).floatValue(),
                !Boolean.FALSE.equals(c.get("tta_diff")),
                Boolean.TRUE.equals(c.get("tta_sign_flip")),
                recipe, heads, slstmHeads < 0 ? heads : slstmHeads, variateHeads,
                // RMS-normalise q and k in the variate attention (off in the GIFT-Eval checkpoints)
                !Boolean.FALSE.equals(c.get("use_qk_norm")));
    }

    private static int same(int current, int value, String what) {
        if (current >= 0 && current != value) {
            throw new IllegalArgumentException("templates disagree on " + what + ": " + current + " vs " + value);
        }
        return value;
    }

    private static int intOf(Map<String, Object> m, String key) {
        Object v = m.get(key);
        if (!(v instanceof Number n) || n.doubleValue() != n.intValue()) {
            throw new IllegalArgumentException("config key '" + key + "' missing or not an integer");
        }
        return n.intValue();
    }

    private static Map<String, Object> map(Map<String, Object> m, String key) {
        Map<String, Object> v = optionalMap(m, key);
        if (v == null) {
            throw new IllegalArgumentException("config key '" + key + "' missing");
        }
        return v;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> optionalMap(Map<String, Object> m, String key) {
        Object v = m.get(key);
        if (v != null && !(v instanceof Map)) {
            throw new IllegalArgumentException("config key '" + key + "' is not an object");
        }
        return (Map<String, Object>) v;
    }
}
