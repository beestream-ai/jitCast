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

    @Override
    default void close() {}
}
