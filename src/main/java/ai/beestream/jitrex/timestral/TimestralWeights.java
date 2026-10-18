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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import ai.beestream.jitrex.io.SafeTensors;
import ai.beestream.jitrex.io.Tensor;

/**
 * The Timestral parameters, bound by name and shape from a checkpoint directory holding
 * {@code config.json} and {@code model.safetensors}, as Mistral publishes them. Every tensor must be
 * bound exactly once, so a checkpoint with extra or missing weights fails at load time.
 *
 * <p>Tensors are views over the file mapping. Two layouts differ from what the engines compute
 * with, and {@link #qkv} and {@link #mlpUp} return rearranged copies: the fused q|k|v projection is
 * stored head-dim-major, and the MLP's fused input projection puts the SiLU half second.
 */
public final class TimestralWeights implements AutoCloseable {

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

    /**
     * One attention sublayer: its pre-norm, the fused {@code q|k|v} projection, the sigmoid gate
     * (null without one) and the output projection.
     */
    public record Attention(boolean time, Tensor norm, Linear wqkv, Linear gate, Linear wo) {}

    /** One transformer layer: its attention sublayers in order, then the SwiGLU MLP. */
    public record Layer(List<Attention> attentions, Tensor mlpNorm, Linear w12, Linear w3) {}

    private final SafeTensors file;
    private final TimestralConfig config;
    private final ResidualBlock patchEmbedding;
    private final Tensor forecastQuery;
    private final List<Layer> layers;
    private final Tensor outNorm;
    private final ResidualBlock head;

    private TimestralWeights(SafeTensors file, TimestralConfig config, ResidualBlock patchEmbedding, Tensor forecastQuery,
            List<Layer> layers, Tensor outNorm, ResidualBlock head) {
        this.file = file;
        this.config = config;
        this.patchEmbedding = patchEmbedding;
        this.forecastQuery = forecastQuery;
        this.layers = layers;
        this.outNorm = outNorm;
        this.head = head;
    }

    /** Loads {@code <dir>/config.json} and {@code <dir>/model.safetensors}. */
    public static TimestralWeights load(Path dir) throws IOException {
        TimestralConfig c = TimestralConfig.load(dir.resolve("config.json"));
        SafeTensors file = SafeTensors.open(dir.resolve("model.safetensors"));
        try {
            Binder b = new Binder(file);
            int d = c.dModel(), p = c.patchSize();
            ResidualBlock embed = b.residual("representation.patch_proj", 2 * p, 4 * d, d, true);
            Tensor query = b.take("representation.forecast_query", d);
            List<Layer> layers = new ArrayList<>();
            for (int i = 0; i < c.layers().size(); i++) {
                String s = "transformer.layers." + i;
                boolean[] axes = c.layers().get(i);
                List<Attention> attns = new ArrayList<>();
                for (int a = 0; a < axes.length; a++) {
                    boolean gated = axes[a] ? c.timeAttnGate() : c.variateAttnGate();
                    attns.add(new Attention(axes[a],
                            b.take(s + ".attn_norms." + a + ".scale", d),
                            b.linear(s + ".attns." + a + ".wQKV", d, 3 * d),
                            gated ? b.linear(s + ".attns." + a + "._gate", d, d) : null,
                            b.linear(s + ".attns." + a + ".wO", d, d)));
                }
                layers.add(new Layer(List.copyOf(attns), b.take(s + ".mlp_norm.scale", d),
                        b.linear(s + ".mlp.w12", d, 2 * c.mlpHiddenDim()), b.linear(s + ".mlp.w3", c.mlpHiddenDim(), d)));
            }
            Tensor outNorm = b.take("transformer.out_norm.scale", d);
            ResidualBlock head = b.residual("head.projection", d, 4 * d, p * c.numQuantiles(), false);
            b.requireAllBound();
            return new TimestralWeights(file, c, embed, query, List.copyOf(layers), outNorm, head);
        } catch (RuntimeException e) {
            file.close();
            throw e;
        }
    }

    public TimestralConfig config() {
        return config;
    }

    public ResidualBlock patchEmbedding() {
        return patchEmbedding;
    }

    public Tensor forecastQuery() {
        return forecastQuery;
    }

    public List<Layer> layers() {
        return layers;
    }

    public Tensor outNorm() {
        return outNorm;
    }

    public ResidualBlock head() {
        return head;
    }

    /** Total parameters bound (each tensor once). */
    public long parameterCount() {
        return file.tensors().values().stream().mapToLong(Tensor::elements).sum();
    }

    /**
     * The fused {@code q|k|v} weight {@code [3·D, D]} with its rows in the heads-major order the
     * attention output uses: row {@code (s·heads + h)·dh + e} is stored row {@code s·D + e·heads + h}.
     */
    public float[] qkv(Attention a) {
        int d = config.dModel(), heads = config.numHeads(), dh = config.headDim();
        float[] src = a.wqkv().weight().toFloatArray();
        float[] dst = new float[src.length];
        for (int s = 0; s < 3; s++) {
            for (int h = 0; h < heads; h++) {
                for (int e = 0; e < dh; e++) {
                    System.arraycopy(src, (s * d + e * heads + h) * d, dst, (s * d + h * dh + e) * d, d);
                }
            }
        }
        return dst;
    }

    /**
     * The MLP's fused input weight {@code [2·F, D]} with its halves swapped, so the SiLU half comes
     * first: the reference computes {@code gate · SiLU(up)} with {@code gate, up = w12(x).chunk(2)},
     * and the shared SwiGLU kernel computes {@code SiLU(first) · second}.
     */
    public float[] mlpUp(Layer l) {
        int d = config.dModel(), f = config.mlpHiddenDim();
        float[] src = l.w12().weight().toFloatArray();
        float[] dst = new float[src.length];
        System.arraycopy(src, f * d, dst, 0, f * d);
        System.arraycopy(src, 0, dst, f * d, f * d);
        return dst;
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
            if (!file.contains(name)) {
                throw new IllegalArgumentException("checkpoint has no tensor " + name);
            }
            return file.get(name).requireF32(shape);
        }

        Linear linear(String prefix, int in, int out) {
            return new Linear(take(prefix + ".weight", out, in), null);
        }

        Linear linear(String prefix, int in, int out, boolean bias) {
            return new Linear(take(prefix + ".weight", out, in), bias ? take(prefix + ".bias", out) : null);
        }

        ResidualBlock residual(String prefix, int in, int hidden, int out, boolean bias) {
            return new ResidualBlock(
                    linear(prefix + ".hidden_layer", in, hidden, bias),
                    linear(prefix + ".output_layer", hidden, out, bias),
                    linear(prefix + ".residual_layer", in, out, bias));
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
