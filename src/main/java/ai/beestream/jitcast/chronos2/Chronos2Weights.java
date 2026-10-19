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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;

/**
 * The Chronos-2 parameters, bound by name and shape from a checkpoint directory holding
 * {@code config.json} and {@code model.safetensors}, as Amazon publishes them. Every tensor must be
 * bound exactly once, so a checkpoint with extra or missing weights fails at load time. Tensors
 * are views over the file mapping.
 */
public final class Chronos2Weights implements AutoCloseable {

    /** {@code y = x·Wᵀ + b}; {@code weight} is {@code [out, in]}, {@code bias} is {@code [out]} or null. */
    public record Linear(Tensor weight, Tensor bias) {
        public int in() {
            return weight.dim(1);
        }

        public int out() {
            return weight.dim(0);
        }
    }

    /** {@code output(ReLU(hidden(x))) + residual(x)}. */
    public record ResidualBlock(Linear hidden, Linear output, Linear residual) {}

    /** One attention sublayer: its pre-norm and the q, k, v and output projections (no biases). */
    public record Attention(Tensor norm, Linear q, Linear k, Linear v, Linear o) {}

    /** One encoder block: time attention, group attention, then the ReLU feed-forward layer. */
    public record Layer(Attention time, Attention group, Tensor mlpNorm, Linear wi, Linear wo) {}

    private final SafeTensors file;
    private final Chronos2Config config;
    private final Tensor tokens;
    private final ResidualBlock inputEmbedding;
    private final List<Layer> layers;
    private final Tensor finalNorm;
    private final ResidualBlock outputEmbedding;

    private Chronos2Weights(SafeTensors file, Chronos2Config config, Tensor tokens, ResidualBlock inputEmbedding,
            List<Layer> layers, Tensor finalNorm, ResidualBlock outputEmbedding) {
        this.file = file;
        this.config = config;
        this.tokens = tokens;
        this.inputEmbedding = inputEmbedding;
        this.layers = layers;
        this.finalNorm = finalNorm;
        this.outputEmbedding = outputEmbedding;
    }

    /** Loads {@code <dir>/config.json} and {@code <dir>/model.safetensors}. */
    public static Chronos2Weights load(Path dir) throws IOException {
        Chronos2Config c = Chronos2Config.load(dir.resolve("config.json"));
        SafeTensors file = SafeTensors.open(dir.resolve("model.safetensors"));
        try {
            Binder b = new Binder(file);
            int d = c.dModel(), inner = c.inner(), ff = c.dFf(), p = c.patchSize();
            Tensor tokens = b.take("shared.weight", c.vocabSize(), d);
            ResidualBlock in = b.residual("input_patch_embedding", 3 * p, ff, d);
            List<Layer> layers = new ArrayList<>();
            for (int i = 0; i < c.numLayers(); i++) {
                String s = "encoder.block." + i + ".layer.";
                layers.add(new Layer(b.attention(s + "0", d, inner), b.attention(s + "1", d, inner),
                        b.take(s + "2.layer_norm.weight", d), b.linear(s + "2.mlp.wi", d, ff, false),
                        b.linear(s + "2.mlp.wo", ff, d, false)));
            }
            Tensor finalNorm = b.take("encoder.final_layer_norm.weight", d);
            ResidualBlock out = b.residual("output_patch_embedding", d, ff, c.numQuantiles() * p);
            b.requireAllBound();
            return new Chronos2Weights(file, c, tokens, in, List.copyOf(layers), finalNorm, out);
        } catch (RuntimeException e) {
            file.close();
            throw e;
        }
    }

    public Chronos2Config config() {
        return config;
    }

    /** {@code shared}: the learned token embeddings {@code [vocab, D]}; row 1 is {@code [REG]}. */
    public Tensor tokens() {
        return tokens;
    }

    public ResidualBlock inputEmbedding() {
        return inputEmbedding;
    }

    public List<Layer> layers() {
        return layers;
    }

    public Tensor finalNorm() {
        return finalNorm;
    }

    public ResidualBlock outputEmbedding() {
        return outputEmbedding;
    }

    /** Total parameters bound (each tensor once). */
    public long parameterCount() {
        return file.tensors().values().stream().mapToLong(Tensor::elements).sum();
    }

    /** The fused {@code q|k|v} weight {@code [3·inner, D]} of an attention sublayer. */
    public float[] qkv(Attention a) {
        float[] q = a.q().weight().toFloatArray(), k = a.k().weight().toFloatArray(), v = a.v().weight().toFloatArray();
        float[] w = new float[q.length + k.length + v.length];
        System.arraycopy(q, 0, w, 0, q.length);
        System.arraycopy(k, 0, w, q.length, k.length);
        System.arraycopy(v, 0, w, q.length + k.length, v.length);
        return w;
    }

    /** The {@code [REG]} token's embedding. */
    public float[] regToken() {
        if (!config.useRegToken()) {
            throw new IllegalStateException("this checkpoint has no [REG] token");
        }
        float[] t = tokens.toFloatArray();
        float[] r = new float[config.dModel()];
        System.arraycopy(t, config.dModel(), r, 0, r.length);
        return r;
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

        Linear linear(String prefix, int in, int out, boolean bias) {
            return new Linear(take(prefix + ".weight", out, in), bias ? take(prefix + ".bias", out) : null);
        }

        Attention attention(String prefix, int d, int inner) {
            String a = prefix + ".self_attention.";
            return new Attention(take(prefix + ".layer_norm.weight", d), linear(a + "q", d, inner, false),
                    linear(a + "k", d, inner, false), linear(a + "v", d, inner, false), linear(a + "o", inner, d, false));
        }

        ResidualBlock residual(String prefix, int in, int hidden, int out) {
            return new ResidualBlock(
                    linear(prefix + ".hidden_layer", in, hidden, true),
                    linear(prefix + ".output_layer", hidden, out, true),
                    linear(prefix + ".residual_layer", in, out, true));
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
