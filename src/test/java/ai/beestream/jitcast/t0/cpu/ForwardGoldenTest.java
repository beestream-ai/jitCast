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
package ai.beestream.jitcast.t0.cpu;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import ai.beestream.jitcast.io.SafeTensors;
import ai.beestream.jitcast.t0.T0Batch;
import ai.beestream.jitcast.t0.T0Config;
import ai.beestream.jitcast.t0.T0Forward;
import ai.beestream.jitcast.t0.T0Golden;
import ai.beestream.jitcast.t0.T0Scaler;
import ai.beestream.jitcast.t0.T0Weights;

/**
 * The scaler and the CPU forward pass against the reference, stage by stage (the scaled window and its
 * statistics, the patch encoder, every layer, the final norm, the decoder and the quantile head), for every
 * recorded forward call of every golden case. The forward pass runs on the reference's scaled window, so
 * each stage is measured on its own; activations are compared on the patches that are not all padding.
 *
 * <p>A stage passes within {@code 1e-4} of the reference, or within {@link #NOISE_FACTOR} times the distance
 * between the reference's own float32 and float64 runs of that call, when that is larger.
 */
public class ForwardGoldenTest {

    private static final double TOLERANCE = 1e-4;
    private static final double NOISE_FACTOR = 4;

    @Test
    public void everyStageMatchesTheReference() throws IOException {
        List<String> failures = new ArrayList<>();
        try (T0Weights weights = T0Weights.load(T0Golden.checkpoint())) {
            CpuT0 ref = new CpuT0(weights);
            T0Config c = weights.config();
            T0Scaler scaler = new T0Scaler(c);
            int p = c.patchSize(), d = c.embedDim(), q = c.quantiles();
            for (String testCase : T0Golden.cases()) {
                try (SafeTensors g = T0Golden.open(testCase)) {
                    for (int k = 0; k < T0Golden.calls(g); k++) {
                        String prefix = "call" + k + ".";
                        T0Batch w = T0Golden.window(g, k);
                        boolean[] cells = new boolean[w.rows() * w.width()];
                        for (int i = 0; i < cells.length; i++) {
                            cells[i] = w.group()[i] >= 0;
                        }
                        T0Scaler.Scaled s = scaler.scale(w);
                        double worst = 0;
                        String worstStage = "";
                        List<String> names = new ArrayList<>(List.of("scaled", "loc", "scale"));
                        List<float[]> values = new ArrayList<>(List.of(s.scaled(), s.loc(), s.scale()));
                        List<boolean[]> keeps = new ArrayList<>(List.of(cells, cells, cells));
                        float[] goldenScaled = g.get(prefix + "scaled").toFloatArray();
                        T0Forward.Input in = T0Forward.Input.of(w, goldenScaled, p);
                        long t0 = System.nanoTime();
                        CpuT0.Stages st = ref.stages(in);
                        long ms = (System.nanoTime() - t0) / 1_000_000;
                        boolean[] patches = in.patchValid();
                        names.add("embed");
                        values.add(st.embed());
                        keeps.add(T0Golden.widen(patches, d));
                        for (int i = 0; i < st.layers().size(); i++) {
                            names.add("layer" + i);
                            values.add(st.layers().get(i));
                            keeps.add(T0Golden.widen(patches, d));
                        }
                        names.add("hidden");
                        values.add(st.hidden());
                        keeps.add(T0Golden.widen(patches, d));
                        names.add("decoded");
                        values.add(st.decoded());
                        keeps.add(T0Golden.widen(patches, p * q));
                        names.add("head");
                        values.add(st.head());
                        keeps.add(T0Golden.widen(patches, p * q));
                        for (int i = 0; i < names.size(); i++) {
                            String name = names.get(i);
                            double err = T0Golden.relativeMaxError(values.get(i), g.get(prefix + name).toFloatArray(), keeps.get(i));
                            if (err > worst) {
                                worst = err;
                                worstStage = name;
                            }
                            double limit = Math.max(TOLERANCE, NOISE_FACTOR * T0Golden.noise(g, prefix + "fp32err." + name));
                            if (!(err <= limit)) {
                                failures.add(testCase + " call" + k + " " + name + ": " + err + " > " + limit);
                            }
                        }
                        System.out.printf("%s call%d rows=%d patches=%d worst %.2e (%s)  %d ms%n", testCase, k, in.rows(),
                                in.patches(), worst, worstStage, ms);
                    }
                }
            }
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }
}
