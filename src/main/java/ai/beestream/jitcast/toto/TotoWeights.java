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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.io.Tensor;

/**
 * A Toto 2.0 checkpoint ({@code config.json} and {@code model.safetensors}), bound by name. The tensors stay
 * memory-mapped until {@link #close()}.
 *
 * <p>The checkpoint is trained with unit scaling (u-μP): every linear layer's output is scaled by
 * {@code 1/√fan_in} ({@code 1/fan_in} for the output head's readouts), and the residual stream is mixed
 * with per-layer weights {@code τ}. The engines apply those constants; the weights here are as stored.
 */
public final class TotoWeights implements AutoCloseable {

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
     * A residual MLP: {@code (linear2(silu(linear1(x))) + skip(x)) / √2}.
     *
     * @param linear1 the hidden layer
     * @param linear2 the output layer
     * @param skip the skip projection
     */
    public record ResidualMlp(Linear linear1, Linear linear2, Linear skip) {}

    /**
     * One transformer layer.
     *
     * @param inProj the fused query, key and value projection, {@code [3·d, d]} with bias
     * @param perDimScale the query's per-dimension scale parameter, {@code [head_dim]}
     * @param outProj the attention output projection, with bias
     * @param attnTau the attention branch's residual weight
     * @param fc1 the gated feed-forward input, {@code [2·d_ff, d]}: the gate rows, then the SiLU rows
     * @param fc2 the feed-forward output, {@code [d, d_ff]}
     * @param mlpTau the feed-forward branch's residual weight
     */
    public record Layer(Linear inProj, Tensor perDimScale, Linear outProj, float attnTau, Linear fc1, Linear fc2, float mlpTau) {}

    private final SafeTensors file;
    private final TotoConfig config;
    private final ResidualMlp patchProj;
    private final List<Layer> layers;
    private final ResidualMlp head;

    private TotoWeights(SafeTensors file, TotoConfig config, ResidualMlp patchProj, List<Layer> layers, ResidualMlp head) {
        this.file = file;
        this.config = config;
        this.patchProj = patchProj;
        this.layers = layers;
        this.head = head;
    }

    /**
     * {@return the checkpoint in a directory}
     *
     * @param dir a directory holding {@code config.json} and {@code model.safetensors}
     * @throws IOException if a file cannot be read
     * @throws IllegalArgumentException if the configuration is unsupported or the tensors do not match it
     */
    public static TotoWeights load(Path dir) throws IOException {
        TotoConfig c = TotoConfig.load(dir.resolve("config.json"));
        SafeTensors file = SafeTensors.open(dir.resolve("model.safetensors"));
        try {
            Binder b = new Binder(file);
            int d = c.dModel(), p = c.patchSize(), hd = c.headDim(), ff = c.dFf();
            ResidualMlp patch = b.residual("patch_proj", 2 * p, 4 * d, d);
            List<Layer> layers = new ArrayList<>();
            for (int i = 0; i < c.numLayers(); i++) {
                String s = "transformer.layers." + i + ".";
                layers.add(new Layer(b.linear(s + "attn.in_proj", d, 3 * d, true), b.take(s + "attn._pds.per_dim_scale", hd),
                        b.linear(s + "attn.out_proj", d, d, true), b.scalar(s + "attn_tau"), b.linear(s + "ffn.fc1", d, 2 * ff, false),
                        b.linear(s + "ffn.fc2", ff, d, false), b.scalar(s + "mlp_tau")));
            }
            ResidualMlp head = b.residual("output_head.param_projection.proj", d, 4 * d, p * TotoConfig.QUANTILES.length);
            b.requireAllBound();
            return new TotoWeights(file, c, patch, List.copyOf(layers), head);
        } catch (RuntimeException e) {
            file.close();
            throw e;
        }
    }

    /** {@return the configuration} */
    public TotoConfig config() {
        return config;
    }

    /** {@return the patch embedding, from {@code 2·patch} inputs (values, then the missing flags) to {@code d}} */
    public ResidualMlp patchProj() {
        return patchProj;
    }

    /** {@return the transformer layers} */
    public List<Layer> layers() {
        return layers;
    }

    /** {@return the output head, from {@code d} to {@code patch × quantiles} (step-major)} */
    public ResidualMlp head() {
        return head;
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

        float scalar(String name) {
            return take(name).toFloatArray()[0];
        }

        Linear linear(String prefix, int in, int out, boolean bias) {
            return new Linear(take(prefix + ".weight", out, in), bias ? take(prefix + ".bias", out) : null);
        }

        ResidualMlp residual(String prefix, int in, int hidden, int out) {
            return new ResidualMlp(linear(prefix + ".linear1", in, hidden, true), linear(prefix + ".linear2", hidden, out, true),
                    linear(prefix + ".skip_proj", in, out, true));
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
