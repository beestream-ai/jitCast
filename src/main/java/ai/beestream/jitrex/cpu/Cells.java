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
package ai.beestream.jitrex.cpu;

import java.util.stream.IntStream;

import ai.beestream.jitrex.model.TiRex2Config;

/**
 * The two recurrent time-mixer cells of the CPU reference. Both map {@code [rows · S · D]} to
 * {@code [rows · S · D]}, processing each row's sequence independently.
 */
public final class Cells {

    private Cells() {}

    /**
     * mLSTM in its stabilised parallel form (equal to xlstm's chunkwise kernel). Per row and head,
     * with {@code F[t] = Σ_{u≤t} logσ(f_u)}:
     *
     * <pre>
     *   logD[t,s] = F[t] − F[s] + i[s]            (s ≤ t)
     *   m[t]      = max(F[t], max_s logD[t,s])    (F[t] is the zero initial state's stabiliser)
     *   C[t,s]    = (q_t·k_s)/√dqk · exp(logD[t,s] − m[t])
     *   h_t       = Σ_s C[t,s]·v_s / (max(|Σ_s C[t,s]|, exp(−m[t])) + eps)
     * </pre>
     *
     * then a per-head LayerNorm, {@code σ(o) ⊙ h} and {@code outProj}.
     */
    static float[] mlstm(float[] x, int rows, int seq, int d, int heads, CpuWeights.Mlstm w) {
        int n = rows * seq;
        CpuWeights.Linear[] parts = {w.q(), w.k(), w.v(), w.o(), w.igate(), w.fgate()};
        int stride = 0;
        for (CpuWeights.Linear l : parts) {
            stride += l.out();
        }
        // the fused projection layout the device kernel reads: q | k | v | o | i | f per token
        float[] p = new float[n * stride];
        int col = 0;
        for (CpuWeights.Linear l : parts) {
            float[] y = l.apply(x, n);
            for (int t = 0; t < n; t++) {
                System.arraycopy(y, t * l.out(), p, t * stride + col, l.out());
            }
            col += l.out();
        }
        float[] h = mlstmCore(p, rows, seq, heads, stride);
        int dv = w.v().out() / heads;
        float[] hn = Ops.headLayerNorm(h, n, heads, dv, w.headNorm(), TiRex2Config.NORM_EPS);
        int oOff = w.q().out() + w.k().out() + w.v().out();
        for (int t = 0; t < n; t++) {
            for (int j = 0; j < d; j++) {
                hn[t * d + j] = (float) (Ops.sigmoid(p[t * stride + oOff + j]) * hn[t * d + j]);
            }
        }
        return w.outProj().apply(hn, n);
    }

    /**
     * The mLSTM recurrence core on the fused projection {@code p [(r·seq + t)·stride]} laid out as
     * {@code q (heads·64) | k (heads·64) | v (heads·128) | o (heads·128) | i (heads) | f (heads)};
     * returns the un-normalised {@code h [(r·seq + t)·heads·128]}.
     */
    public static float[] mlstmCore(float[] p, int rows, int seq, int heads, int stride) {
        int dqk = 64, dv = 128;
        int kOff = heads * dqk, vOff = 2 * heads * dqk, iOff = vOff + 2 * heads * dv, fOff = iOff + heads;
        double scale = 1.0 / Math.sqrt(dqk);
        float[] h = new float[rows * seq * heads * dv];
        IntStream.range(0, rows * heads).parallel().forEach(rh -> {
            int r = rh / heads, hd = rh % heads;
            double[] cumF = new double[seq];
            double[] in = new double[seq];
            double acc = 0;
            for (int t = 0; t < seq; t++) {
                int base = (r * seq + t) * stride;
                acc += Ops.logSigmoid(Ops.softCap(p[base + fOff + hd], TiRex2Config.GATE_SOFT_CAP));
                cumF[t] = acc;
                in[t] = Ops.softCap(p[base + iOff + hd], TiRex2Config.GATE_SOFT_CAP);
            }
            double[] num = new double[dv];
            for (int t = 0; t < seq; t++) {
                double m = cumF[t];
                for (int s = 0; s <= t; s++) {
                    m = Math.max(m, cumF[t] - cumF[s] + in[s]);
                }
                int qo = (r * seq + t) * stride + hd * dqk;
                double den = 0;
                java.util.Arrays.fill(num, 0);
                for (int s = 0; s <= t; s++) {
                    int so = (r * seq + s) * stride;
                    double dot = 0;
                    for (int j = 0; j < dqk; j++) {
                        dot += (double) p[qo + j] * p[so + kOff + hd * dqk + j];
                    }
                    double cts = dot * scale * Math.exp(cumF[t] - cumF[s] + in[s] - m);
                    den += cts;
                    for (int j = 0; j < dv; j++) {
                        num[j] += cts * p[so + vOff + hd * dv + j];
                    }
                }
                double norm = Math.max(Math.abs(den), Math.exp(-m)) + TiRex2Config.MLSTM_EPS;
                int ho = (r * seq + t) * heads * dv + hd * dv;
                for (int j = 0; j < dv; j++) {
                    h[ho + j] = (float) (num[j] / norm);
                }
            }
        });
        return h;
    }

    /**
     * sLSTM recurrence (FlashRNN's vanilla backend). Gate slots are {@code (i, f, z, o)}; states
     * {@code y, c, n, m} start at zero. Per step, with {@code raw = Wx + R·y + b}:
     *
     * <pre>
     *   logf = logσ(f) + m
     *   m'   = t == 0 ? i : max(i, logf)
     *   c'   = e^(logf−m')·c + e^(i−m')·tanh(z)
     *   n'   = max(e^(logf−m')·n + e^(i−m'), 1)
     *   y'   = σ(o)·c'/n'
     * </pre>
     *
     * FlashRNN's condition is "all of n is zero", which after the first step can never hold again
     * because {@code n' ≥ 1}; so it is exactly "t == 0". The output is a per-head LayerNorm of y.
     */
    static float[] slstm(float[] x, int rows, int seq, int d, int heads, CpuWeights.Slstm w) {
        int n = rows * seq;
        float[] wx = headwiseGates(x, n, d, heads, w.gates());
        float[] y = slstmRecurrence(wx, rows, seq, d, heads, w.r(), w.b());
        return Ops.headLayerNorm(y, n, heads, d / heads, w.groupNorm(), TiRex2Config.NORM_EPS);
    }

    /**
     * The four headwise gate projections, laid out as the device kernel writes them:
     * {@code wx[rt·4d + g·d + h·dh + o] = Σ_j x[rt·d + h·dh + j] · gates[g][(h·dh + o)·dh + j]}.
     */
    public static float[] headwiseGates(float[] x, int n, int d, int heads, float[][] gates) {
        int dh = d / heads;
        float[] wx = new float[n * 4 * d];
        IntStream.range(0, n).parallel().forEach(rt -> {
            int xo = rt * d;
            for (int g = 0; g < 4; g++) {
                float[] wg = gates[g];
                for (int h = 0; h < heads; h++) {
                    for (int o = 0; o < dh; o++) {
                        int wo = (h * dh + o) * dh;
                        double acc = 0;
                        for (int j = 0; j < dh; j++) {
                            acc += (double) x[xo + h * dh + j] * wg[wo + j];
                        }
                        wx[rt * 4 * d + g * d + h * dh + o] = (float) acc;
                    }
                }
            }
        });
        return wx;
    }

    /** The recurrence over the gate pre-activations {@code wx} (layout of {@link #headwiseGates}); returns y. */
    public static float[] slstmRecurrence(float[] wx, int rows, int seq, int d, int heads, float[] r, float[] b) {
        int dh = d / heads;
        float[] out = new float[rows * seq * d];
        IntStream.range(0, rows).parallel().forEach(row -> {
            double[] y = new double[d], c = new double[d], nn = new double[d], m = new double[d];
            double[] raw = new double[4 * d];
            for (int t = 0; t < seq; t++) {
                int base = (row * seq + t) * d;
                int wxBase = (row * seq + t) * 4 * d;
                // raw[g·d + h·dh + e] = Wx + Σ_p y[h·dh + p]·R[h][p][g][e] + b[h][g][e]
                for (int h = 0; h < heads; h++) {
                    for (int g = 0; g < 4; g++) {
                        for (int e = 0; e < dh; e++) {
                            raw[g * d + h * dh + e] = wx[wxBase + g * d + h * dh + e] + b[(h * 4 + g) * dh + e];
                        }
                    }
                    for (int p = 0; p < dh; p++) {
                        double yp = y[h * dh + p];
                        if (yp == 0) {
                            continue;
                        }
                        int ro = (h * dh + p) * 4 * dh;
                        for (int g = 0; g < 4; g++) {
                            for (int e = 0; e < dh; e++) {
                                raw[g * d + h * dh + e] += yp * r[ro + g * dh + e];
                            }
                        }
                    }
                }
                for (int j = 0; j < d; j++) {
                    double iRaw = raw[j], fRaw = raw[d + j], zRaw = raw[2 * d + j], oRaw = raw[3 * d + j];
                    double logf = Ops.logSigmoid(fRaw) + m[j];
                    double mNew = t == 0 ? iRaw : Math.max(iRaw, logf);
                    double iG = Math.exp(iRaw - mNew);
                    double fG = Math.exp(logf - mNew);
                    c[j] = fG * c[j] + iG * Math.tanh(zRaw);
                    nn[j] = Math.max(fG * nn[j] + iG, 1.0);
                    y[j] = Ops.sigmoid(oRaw) * c[j] / nn[j];
                    m[j] = mNew;
                    out[base + j] = (float) y[j];
                }
            }
        });
        return out;
    }
}
