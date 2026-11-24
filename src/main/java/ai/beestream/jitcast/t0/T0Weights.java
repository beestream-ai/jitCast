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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;

/**
 * Every tensor of a t0 checkpoint, bound by name and shape; unknown or missing tensors fail the load. The
 * tensors stay memory-mapped until {@link #close()}.
 */
public final class T0Weights implements AutoCloseable {

    /**
     * A linear layer.
     *
     * @param weight {@code [out, in]}
     * @param bias {@code [out]}
     */
    public record Linear(Tensor weight, Tensor bias) {
        /** {@return the input width} */
        public int in() {
            return weight.dim(1);
        }

        /** {@return the output width} */
        public int out() {
            return weight.dim(0);
        }
    }

    /**
     * {@code output(relu(hidden(x))) + residual(x)}.
     *
     * @param hidden the first layer
     * @param output the second layer
     * @param residual the linear shortcut
     */
    public record ResidualBlock(Linear hidden, Linear output, Linear residual) {}

    /**
     * One transformer layer: pre-norm attention (across time with the rotary embedding, or across the
     * variates of a series), then a pre-norm SwiGLU feed-forward, each with a residual connection.
     *
     * @param group whether the layer attends across variates
     * @param attnNorm the RMSNorm scale before attention, {@code [D]}
     * @param qkv the fused query, key and value projection, {@code [3 · D, D]}
     * @param queryNorm the per-head RMSNorm scale of the queries, {@code [D / heads]}
     * @param keyNorm the per-head RMSNorm scale of the keys, {@code [D / heads]}
     * @param out the attention output projection
     * @param mlpNorm the RMSNorm scale before the feed-forward, {@code [D]}
     * @param up the feed-forward input layer, {@code [2 · hidden, D]}: the gate, then the value
     * @param down the feed-forward output layer, {@code [D, hidden]}
     */
    public record Layer(boolean group, Tensor attnNorm, Linear qkv, Tensor queryNorm, Tensor keyNorm, Linear out,
            Tensor mlpNorm, Linear up, Linear down) {}

    private final SafeTensors file;
    private final T0Config config;
    private final ResidualBlock patchEncoder;
    private final Tensor typeEmbeddings;
    private final List<Layer> layers;
    private final Tensor outNorm;
    private final ResidualBlock decoder;

    private T0Weights(SafeTensors file, T0Config config, ResidualBlock patchEncoder, Tensor typeEmbeddings,
            List<Layer> layers, Tensor outNorm, ResidualBlock decoder) {
        this.file = file;
        this.config = config;
        this.patchEncoder = patchEncoder;
        this.typeEmbeddings = typeEmbeddings;
        this.layers = layers;
        this.outNorm = outNorm;
        this.decoder = decoder;
    }

    /**
     * Loads {@code <dir>/model.safetensors} with {@code <dir>/config.json}.
     *
     * @param dir the checkpoint directory
     * @return the weights; close them when done
     * @throws IOException if a file cannot be read
     * @throws IllegalArgumentException if the configuration or a tensor is not what the engine supports
     */
    public static T0Weights load(Path dir) throws IOException {
        T0Config c = T0Config.load(dir.resolve("config.json"));
        SafeTensors file = SafeTensors.open(dir.resolve("model.safetensors"));
        try {
            Binder b = new Binder(file);
            int d = c.embedDim(), p = c.patchSize(), hd = c.headDim(), ff = c.mlpHiddenDim(), q = c.quantiles();
            ResidualBlock encoder = b.residual("patch_encoder.projection", 3 * p, d, d);
            Tensor types = b.take("patch_encoder.type_embeddings.weight", 3, d);
            List<Layer> layers = new ArrayList<>();
            for (int i = 0; i < c.numLayers(); i++) {
                String s = "transformer.layers." + i + ".";
                String a = s + "attention_block.attention.";
                layers.add(new Layer(c.isGroupLayer(i), b.take(s + "attention_block.norm.scale", d),
                        b.linear(a + "wQKV", d, 3 * d), b.take(a + "q_norm.scale", hd), b.take(a + "k_norm.scale", hd),
                        b.linear(a + "wO", d, d), b.take(s + "norm.scale", d), b.linear(s + "mlp.0", d, 2 * ff),
                        b.linear(s + "mlp.2", ff, d)));
            }
            Tensor outNorm = b.take("transformer.out_norm.scale", d);
            ResidualBlock decoder = b.residual("decoder", d, d, p * q);
            Tensor levels = b.take("head.quantile_levels", q);
            for (int i = 0; i < q; i++) {
                if (Math.abs(levels.getFloat(i) - (float) c.quantileLevels()[i]) > 1e-6f) {
                    throw new IllegalArgumentException("head.quantile_levels does not match config.json's quantile_levels");
                }
            }
            b.requireAllBound();
            return new T0Weights(file, c, encoder, types, List.copyOf(layers), outNorm, decoder);
        } catch (RuntimeException e) {
            file.close();
            throw e;
        }
    }

    /** {@return the checkpoint's configuration} */
    public T0Config config() {
        return config;
    }

    /** {@return the patch encoder: a patch's {@code [values ‖ time index ‖ validity]} to an embedding} */
    public ResidualBlock patchEncoder() {
        return patchEncoder;
    }

    /** {@return the type embeddings, {@code [3, D]}: target, historical covariate, known-future covariate} */
    public Tensor typeEmbeddings() {
        return typeEmbeddings;
    }

    /** {@return the transformer layers} */
    public List<Layer> layers() {
        return layers;
    }

    /** {@return the RMSNorm scale after the last layer} */
    public Tensor outNorm() {
        return outNorm;
    }

    /** {@return the decoder: an embedding to {@code patchSize × quantiles} raw values} */
    public ResidualBlock decoder() {
        return decoder;
    }

    /** {@return the number of parameters} */
    public long parameterCount() {
        long n = 0;
        for (Tensor t : file.tensors().values()) {
            n += t.elements();
        }
        return n;
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
            return new Linear(take(prefix + ".weight", out, in), take(prefix + ".bias", out));
        }

        ResidualBlock residual(String prefix, int in, int hidden, int out) {
            return new ResidualBlock(linear(prefix + ".mlp.hidden_layer", in, hidden),
                    linear(prefix + ".mlp.output_layer", hidden, out), linear(prefix + ".residual_layer", in, out));
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
