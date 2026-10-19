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
package ai.beestream.jitcast.tirex2.cpu;

import ai.beestream.jitcast.cpu.Ops;

/**
 * The TiRex-2-specific FP32 operators of the CPU reference: the xlstm head LayerNorm and gate
 * functions, the residual block and the SwiGLU FFN. The shared operators are in {@link Ops}.
 */
public final class TiRexOps {

    /**
     * LayerNorm over each head of each row ({@code cols = heads × headDim}), centred with the biased
     * variance, then scaled by the full-width weight {@code w [cols]} (xlstm MultiHeadLayerNorm).
     */
    public static float[] headLayerNorm(float[] x, int rows, int heads, int headDim, float[] w, float eps) {
        float[] y = new float[x.length];
        for (int r = 0; r < rows; r++) {
            for (int h = 0; h < heads; h++) {
                int o = (r * heads + h) * headDim;
                double mean = 0;
                for (int i = 0; i < headDim; i++) {
                    mean += x[o + i];
                }
                mean /= headDim;
                double var = 0;
                for (int i = 0; i < headDim; i++) {
                    double dv = x[o + i] - mean;
                    var += dv * dv;
                }
                var /= headDim;
                double inv = 1.0 / Math.sqrt(var + eps);
                for (int i = 0; i < headDim; i++) {
                    y[o + i] = (float) ((x[o + i] - mean) * inv) * w[h * headDim + i];
                }
            }
        }
        return y;
    }

    /** {@code log(Ops.sigmoid(z))}, stable for large |z|. */
    public static double logSigmoid(double z) {
        return Math.min(z, 0.0) - Math.log1p(Math.exp(-Math.abs(z)));
    }

    /** {@code cap · tanh(z / cap)}, the xlstm gate soft-cap. */
    public static float softCap(float z, float cap) {
        return (float) (cap * Math.tanh(z / cap));
    }

    /** {@code output(SiLU(hidden(x))) + residual(x)}, the TiRex ResidualBlock without layer norm. */
    static float[] residualBlock(float[] x, int rows, CpuWeights.ResidualBlock w) {
        float[] h = w.hidden().apply(x, rows);
        Ops.siluInPlace(h);
        float[] y = w.output().apply(h, rows);
        Ops.addInPlace(y, w.residual().apply(x, rows));
        return y;
    }

    /** The xlstm-large SwiGLU FFN: {@code down(SiLU(upGate(x)) · up(x))}. */
    static float[] swiGlu(float[] x, int rows, CpuWeights.Linear upGate, CpuWeights.Linear up, CpuWeights.Linear down) {
        float[] g = upGate.apply(x, rows);
        float[] u = up.apply(x, rows);
        for (int i = 0; i < g.length; i++) {
            g[i] = Ops.silu(g[i]) * u[i];
        }
        return down.apply(g, rows);
    }

    private TiRexOps() {}
}
