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

/**
 * One engine's TimesFM forward pass over a batch of patch sequences, returning the heads of each row's
 * last patch. The sequences are causal: a patch attends only to itself and the patches before it, so
 * the last patch's outputs are those of the reference's cached decoding.
 */
public interface PatchForward extends AutoCloseable {

    /**
     * The inputs of one forward pass.
     *
     * @param rows the number of sequences
     * @param patches the patches per sequence
     * @param values {@code [rows · patches · P]} the normalized values, 0 where masked
     * @param masks {@code [rows · patches · P]} 1 where a step is padding, else 0; padding only precedes
     *     the observations
     * @param spread whether to compute the continuous quantile head too
     */
    record Batch(int rows, int patches, float[] values, float[] masks, boolean spread) {}

    /**
     * The heads of each row's last patch, still normalized.
     *
     * @param point {@code [rows · horizonLength · outputs]}
     * @param spread {@code [rows · quantileHorizonLength · outputs]}, or null when not asked for
     */
    record Heads(float[] point, float[] spread) {}

    /**
     * {@return the heads of each row's last patch}
     *
     * @param batch the inputs
     */
    Heads forward(Batch batch);

    /** Supplies a decoding step's new patches from the heads of the step before. */
    @FunctionalInterface
    interface Feeder {
        /**
         * {@return the next patches to append, one batch row per sequence, without padding}
         *
         * @param previous the heads after the prefill (step 0) or the previous step
         * @param step the step about to run, from 0
         */
        Batch next(Heads previous, int step);
    }

    /**
     * Decodes: the prefill, then {@code steps} steps that each append the patches {@code feeder} supplies
     * and return the heads of the new last patch. This default runs every step over the whole sequence;
     * an engine may keep the keys and values of earlier patches instead, with the same results.
     *
     * @param prefill the prefill's inputs
     * @param steps the decoding steps after the prefill
     * @param feeder the source of each step's patches
     * @return the heads of the prefill, then of each step
     */
    default java.util.List<Heads> decode(Batch prefill, int steps, Feeder feeder) {
        java.util.List<Heads> out = new java.util.ArrayList<>();
        Heads h = forward(prefill);
        out.add(h);
        int rows = prefill.rows(), n = prefill.patches();
        int p = prefill.values().length / (rows * n);
        float[] values = prefill.values(), masks = prefill.masks();
        for (int s = 0; s < steps; s++) {
            Batch next = feeder.next(h, s);
            int add = next.patches();
            float[] v = new float[rows * (n + add) * p], mk = new float[v.length];
            for (int r = 0; r < rows; r++) {
                System.arraycopy(values, r * n * p, v, r * (n + add) * p, n * p);
                System.arraycopy(masks, r * n * p, mk, r * (n + add) * p, n * p);
                System.arraycopy(next.values(), r * add * p, v, (r * (n + add) + n) * p, add * p);
                System.arraycopy(next.masks(), r * add * p, mk, (r * (n + add) + n) * p, add * p);
            }
            values = v;
            masks = mk;
            n += add;
            h = forward(new Batch(rows, n, values, masks, false));
            out.add(h);
        }
        return out;
    }

    @Override
    default void close() {}
}
