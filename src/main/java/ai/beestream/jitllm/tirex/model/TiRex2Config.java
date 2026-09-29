package ai.beestream.jitllm.tirex.model;

import java.util.List;
import java.util.Map;

import org.beehive.jitllm.server.Json;

/**
 * The TiRex-2 hyper-parameters, read from the {@code model-config.yaml} that
 * {@code scripts/convert_ckpt.py} embeds as JSON under {@code tirex2.config}.
 *
 * <p>Only the values the forward pass needs are kept. Constants that the reference code hard-codes
 * (not in the config) are here as well, so every number the engine uses has one home.
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

    /** The time mixer of a block. Every block's variate mixer is attention. */
    public enum BlockType {
        MLSTM, SLSTM
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

    public int numPatches() {
        return (contextLen + futureLen) / patchSize;
    }

    public int paddedLength() {
        return contextLen + futureLen;
    }

    public int numQuantiles() {
        return quantiles.length;
    }

    public int mlstmQkDim() {
        return (int) (embeddingDim * QK_DIM_FACTOR);
    }

    public int ffnDim() {
        int raw = (int) (embeddingDim * FFN_PROJ_FACTOR);
        return (raw + FFN_ROUND_TO - 1) / FFN_ROUND_TO * FFN_ROUND_TO;
    }

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
        Map<String, Object> outNorm = map(c, "stack_out_norm_config");
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
        if (!(v instanceof Number n)) {
            throw new IllegalArgumentException("config key '" + key + "' missing or not a number");
        }
        return n.intValue();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Map<String, Object> m, String key) {
        return (Map<String, Object>) m.get(key);
    }
}
