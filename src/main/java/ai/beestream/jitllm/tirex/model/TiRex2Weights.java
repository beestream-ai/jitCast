package ai.beestream.jitllm.tirex.model;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import ai.beestream.jitllm.tirex.io.SafeTensors;
import ai.beestream.jitllm.tirex.io.Tensor;
import ai.beestream.jitllm.tirex.model.TiRex2Config.BlockType;

/**
 * The TiRex-2 parameters, bound by name and shape from the safetensors file that
 * {@code scripts/convert_ckpt.py} writes. Every tensor in the file must be bound exactly once, so a
 * checkpoint with extra or missing weights fails at load time, not as wrong forecasts.
 *
 * <p>Tensors are views over the file mapping; nothing is copied. Linear weights keep PyTorch's
 * {@code [out, in]} layout.
 */
public final class TiRex2Weights implements AutoCloseable {

    /** {@code y = x·Wᵀ + b}; {@code weight} is {@code [out, in]}, {@code bias} is {@code [out]} or null. */
    public record Linear(Tensor weight, Tensor bias) {
        public int in() {
            return weight.dim(1);
        }

        public int out() {
            return weight.dim(0);
        }
    }

    /** {@code output(SiLU(hidden(x))) + residual(x)}. */
    public record ResidualBlock(Linear hidden, Linear output, Linear residual) {}

    public sealed interface Cell permits Mlstm, Slstm {}

    /**
     * mLSTM cell. {@code q, k: D→qk}, {@code v, o: D→D}, {@code igate, fgate: D→heads} (with bias,
     * soft-capped), per-head LayerNorm weight {@code headNorm [D]}, then {@code outProj: D→D}.
     */
    public record Mlstm(Linear q, Linear k, Linear v, Linear o, Linear igate, Linear fgate, Tensor headNorm, Linear outProj)
            implements Cell {}

    /**
     * sLSTM cell (FlashRNN, "HPGD" recurrent / "HGD" bias layouts).
     *
     * <p>{@code gates[g]} is a headwise projection {@code [H, Dh(out), Dh(in)]} and produces the
     * pre-activation of FlashRNN gate slot {@code g}, whose meaning is {@code (i, f, z, o)}. TiRex-2
     * stacks its projections as {@code (fgate, igate, zgate, ogate)}, so slot 0 is the checkpoint's
     * {@code fgate} tensor and feeds the input gate; the names are not the roles.
     *
     * @param gates the four headwise projections in slot order
     * @param recurrent {@code R [H, Dh(prev), 4, Dh]}
     * @param bias {@code b [H, 4, Dh]}
     * @param groupNorm per-head LayerNorm weight {@code [D]}
     */
    public record Slstm(Tensor[] gates, Tensor recurrent, Tensor bias, Tensor groupNorm) implements Cell {}

    /** BiXLSTM: RMSNorm, cell (forward, and reverse over known covariates), combination, SwiGLU FFN. */
    public record TimeMixer(Tensor normLstm, Cell cell, Linear combination, Tensor normFfn,
            Linear ffnUpGate, Linear ffnUp, Linear ffnDown) {}

    /** Attention across variates, then a SiLU MLP; both pre-normalised with RMSNorm and residual. */
    public record VariateMixer(Tensor normAttn, Linear wq, Linear wk, Linear wv, Linear wo,
            Tensor normFfn, Linear ffnIn, Linear ffnOut) {}

    public record Block(BlockType type, TimeMixer time, VariateMixer variate) {}

    private final SafeTensors file;
    private final TiRex2Config config;
    private final ResidualBlock inputEmbedding;
    private final List<Block> blocks;
    private final Tensor stackOutNorm;
    private final ResidualBlock outputEmbedding;

    private TiRex2Weights(SafeTensors file, TiRex2Config config, ResidualBlock inputEmbedding, List<Block> blocks,
            Tensor stackOutNorm, ResidualBlock outputEmbedding) {
        this.file = file;
        this.config = config;
        this.inputEmbedding = inputEmbedding;
        this.blocks = blocks;
        this.stackOutNorm = stackOutNorm;
        this.outputEmbedding = outputEmbedding;
    }

    public static TiRex2Weights load(Path path) throws IOException {
        SafeTensors file = SafeTensors.open(path);
        try {
            String json = file.metadata().get("tirex2.config");
            if (json == null) {
                throw new IOException(path + " has no tirex2.config metadata; convert it with scripts/convert_ckpt.py");
            }
            TiRex2Config c = TiRex2Config.fromJson(json);
            Binder b = new Binder(file);
            int d = c.embeddingDim();
            int p = c.patchSize();

            ResidualBlock in = b.residual("input_patch_embedding", 2 * p, c.inputFfDim(), d);
            List<Block> blocks = new ArrayList<>();
            for (int i = 0; i < c.numBlocks(); i++) {
                blocks.add(b.block(c, i));
            }
            Tensor outNorm = b.vector("stack_out_norm.weight", d);
            ResidualBlock out = b.residual("output_patch_embedding", d, c.outputFfDim(), c.numQuantiles() * p);
            b.requireAllBound();
            return new TiRex2Weights(file, c, in, List.copyOf(blocks), outNorm, out);
        } catch (IOException | RuntimeException e) {
            file.close();
            throw e;
        }
    }

    public TiRex2Config config() {
        return config;
    }

    public ResidualBlock inputEmbedding() {
        return inputEmbedding;
    }

    public List<Block> blocks() {
        return blocks;
    }

    public Tensor stackOutNorm() {
        return stackOutNorm;
    }

    public ResidualBlock outputEmbedding() {
        return outputEmbedding;
    }

    /** Total parameters bound (each tensor once). */
    public long parameterCount() {
        return file.tensors().values().stream().mapToLong(Tensor::elements).sum();
    }

    @Override
    public void close() {
        file.close();
    }

    /** Looks tensors up by name, checks their shapes and records which ones were used. */
    private static final class Binder {
        private final SafeTensors file;
        private final Set<String> bound = new HashSet<>();

        Binder(SafeTensors file) {
            this.file = file;
        }

        Tensor take(String name, long... shape) {
            if (!bound.add(name)) {
                throw new IllegalStateException(name + " bound twice");
            }
            return file.get(name).requireF32(shape);
        }

        Tensor vector(String name, int n) {
            return take(name, n);
        }

        Linear linear(String prefix, int in, int out, boolean bias) {
            return new Linear(take(prefix + ".weight", out, in), bias ? take(prefix + ".bias", out) : null);
        }

        ResidualBlock residual(String prefix, int in, int hidden, int out) {
            return new ResidualBlock(
                    linear(prefix + ".hidden_layer", in, hidden, true),
                    linear(prefix + ".output_layer", hidden, out, true),
                    linear(prefix + ".residual_layer", in, out, true));
        }

        Block block(TiRex2Config c, int i) {
            String s = "stack." + i;
            int d = c.embeddingDim();
            BlockType type = c.recipe().get(i);
            String cell = s + ".time_mixer.fwd_cell";
            Cell cellWeights = switch (type) {
                case MLSTM -> {
                    int qk = c.mlstmQkDim();
                    int h = c.timeMixerHeads();
                    yield new Mlstm(
                            linear(cell + ".q", d, qk, false),
                            linear(cell + ".k", d, qk, false),
                            linear(cell + ".v", d, d, false),
                            linear(cell + ".ogate_preact", d, d, false),
                            linear(cell + ".igate_preact", d, h, true),
                            linear(cell + ".fgate_preact", d, h, true),
                            vector(cell + ".multihead_norm.weight", d),
                            linear(cell + ".out_proj", d, d, false));
                }
                case SLSTM -> {
                    int h = c.slstmHeads();
                    int dh = d / h;
                    Tensor[] gates = new Tensor[4];
                    String[] slotSource = {"fgate", "igate", "zgate", "ogate"};
                    for (int g = 0; g < 4; g++) {
                        gates[g] = take(cell + "." + slotSource[g] + ".weight", h, dh, dh);
                    }
                    yield new Slstm(gates,
                            take(cell + "._recurrent_kernel_", h, dh, 4, dh),
                            take(cell + "._bias_", h, 4, dh),
                            vector(cell + ".group_norm.weight", d));
                }
            };
            String tm = s + ".time_mixer";
            int ffn = c.ffnDim();
            TimeMixer time = new TimeMixer(
                    vector(tm + ".norm_lstm.weight", d),
                    cellWeights,
                    linear(tm + ".combination", 2 * d, d, false),
                    vector(tm + ".norm_ffn.weight", d),
                    linear(tm + ".ffn.proj_up_gate", d, ffn, false),
                    linear(tm + ".ffn.proj_up", d, ffn, false),
                    linear(tm + ".ffn.proj_down", ffn, d, false));
            String vm = s + ".variate_mixer";
            int hidden = TiRex2Config.ATTENTION_FFN_FACTOR * d;
            VariateMixer variate = new VariateMixer(
                    vector(vm + ".norm_attn.weight", d),
                    linear(vm + ".attn.WQ", d, d, false),
                    linear(vm + ".attn.WK", d, d, false),
                    linear(vm + ".attn.WV", d, d, false),
                    linear(vm + ".attn.WO", d, d, false),
                    vector(vm + ".norm_ffn.weight", d),
                    linear(vm + ".ffn.wi", d, hidden, false),
                    linear(vm + ".ffn.wo", hidden, d, false));
            return new Block(type, time, variate);
        }

        void requireAllBound() {
            Set<String> unbound = new TreeSet<>(file.tensors().keySet());
            unbound.removeAll(bound);
            if (!unbound.isEmpty()) {
                throw new IllegalArgumentException("unexpected tensors in checkpoint: " + unbound);
            }
        }
    }
}
