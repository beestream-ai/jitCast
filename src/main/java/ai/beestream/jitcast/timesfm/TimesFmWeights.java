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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;

/**
 * Every tensor of a TimesFM 2.5 checkpoint, bound by name and shape; unknown or missing tensors fail
 * the load. The tensors stay memory-mapped until {@link #close()}.
 */
public final class TimesFmWeights implements AutoCloseable {

    /**
     * A linear layer.
     *
     * @param weight {@code [out, in]}
     * @param bias {@code [out]}, or null
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
     * {@code output(swish(hidden(x))) + residual(x)}.
     *
     * @param hidden the first layer
     * @param output the second layer
     * @param residual the linear shortcut
     */
    public record ResidualBlock(Linear hidden, Linear output, Linear residual) {}

    /**
     * One transformer layer.
     *
     * @param preAttnNorm the RMSNorm scale before attention
     * @param qkv the fused query, key and value projection, {@code [3 · D, D]}
     * @param queryNorm the per-head RMSNorm scale of the queries
     * @param keyNorm the per-head RMSNorm scale of the keys
     * @param perDimScale the learned per-dimension query scale (before its softplus)
     * @param out the attention output projection
     * @param postAttnNorm the RMSNorm scale of the attention output
     * @param preFfNorm the RMSNorm scale before the feed-forward layer
     * @param ff0 the feed-forward input layer
     * @param ff1 the feed-forward output layer
     * @param postFfNorm the RMSNorm scale of the feed-forward output
     */
    public record Layer(Tensor preAttnNorm, Linear qkv, Tensor queryNorm, Tensor keyNorm, Tensor perDimScale, Linear out,
            Tensor postAttnNorm, Tensor preFfNorm, Linear ff0, Linear ff1, Tensor postFfNorm) {}

    private final SafeTensors file;
    private final TimesFmConfig config;
    private final ResidualBlock tokenizer;
    private final List<Layer> layers;
    private final ResidualBlock pointHead;
    private final ResidualBlock quantileHead;

    private TimesFmWeights(SafeTensors file, TimesFmConfig config, ResidualBlock tokenizer, List<Layer> layers,
            ResidualBlock pointHead, ResidualBlock quantileHead) {
        this.file = file;
        this.config = config;
        this.tokenizer = tokenizer;
        this.layers = layers;
        this.pointHead = pointHead;
        this.quantileHead = quantileHead;
    }

    /**
     * Loads {@code <dir>/model.safetensors} with {@code <dir>/config.json}; without a {@code config.json} the
     * published TimesFM 2.5 200M configuration is assumed.
     *
     * @param dir the checkpoint directory
     * @return the weights; close them when done
     * @throws IOException if a file cannot be read
     * @throws IllegalArgumentException if the configuration or a tensor is not what the engine supports
     */
    public static TimesFmWeights load(Path dir) throws IOException {
        Path json = dir.resolve("config.json");
        TimesFmConfig c = Files.exists(json) ? TimesFmConfig.load(json) : TimesFmConfig.fromJson("{\"quantiles\": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}");
        SafeTensors file = SafeTensors.open(dir.resolve("model.safetensors"));
        try {
            Binder b = new Binder(file);
            int d = c.hiddenSize(), p = c.patchLength(), hd = c.headDim(), ff = c.intermediateSize();
            ResidualBlock tokenizer = b.residual("tokenizer", 2 * p, d, d, true);
            List<Layer> layers = new ArrayList<>();
            for (int i = 0; i < c.numLayers(); i++) {
                String s = "stacked_xf." + i + ".";
                layers.add(new Layer(b.take(s + "pre_attn_ln.scale", d), b.linear(s + "attn.qkv_proj", d, 3 * d, false),
                        b.take(s + "attn.query_ln.scale", hd), b.take(s + "attn.key_ln.scale", hd),
                        b.take(s + "attn.per_dim_scale.per_dim_scale", hd), b.linear(s + "attn.out", d, d, false),
                        b.take(s + "post_attn_ln.scale", d), b.take(s + "pre_ff_ln.scale", d),
                        b.linear(s + "ff0", d, ff, false), b.linear(s + "ff1", ff, d, false), b.take(s + "post_ff_ln.scale", d)));
            }
            ResidualBlock point = b.residual("output_projection_point", d, d, c.horizonLength() * c.outputs(), false);
            ResidualBlock quantiles = b.residual("output_projection_quantiles", d, d,
                    c.quantileHorizonLength() * c.outputs(), false);
            b.requireAllBound();
            return new TimesFmWeights(file, c, tokenizer, List.copyOf(layers), point, quantiles);
        } catch (RuntimeException e) {
            file.close();
            throw e;
        }
    }

    /** {@return the checkpoint's configuration} */
    public TimesFmConfig config() {
        return config;
    }

    /** {@return the tokenizer: a patch's values and mask to an embedding} */
    public ResidualBlock tokenizer() {
        return tokenizer;
    }

    /** {@return the transformer layers} */
    public List<Layer> layers() {
        return layers;
    }

    /** {@return the point head: an embedding to {@code horizonLength × outputs} values} */
    public ResidualBlock pointHead() {
        return pointHead;
    }

    /** {@return the continuous quantile head: an embedding to {@code quantileHorizonLength × outputs} values} */
    public ResidualBlock quantileHead() {
        return quantileHead;
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

        Linear linear(String prefix, int in, int out, boolean bias) {
            return new Linear(take(prefix + ".weight", out, in), bias ? take(prefix + ".bias", out) : null);
        }

        ResidualBlock residual(String prefix, int in, int hidden, int out, boolean bias) {
            return new ResidualBlock(linear(prefix + ".hidden_layer", in, hidden, bias),
                    linear(prefix + ".output_layer", hidden, out, bias), linear(prefix + ".residual_layer", in, out, bias));
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
