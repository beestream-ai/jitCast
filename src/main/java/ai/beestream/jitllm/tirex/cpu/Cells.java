package ai.beestream.jitllm.tirex.cpu;

import java.util.stream.IntStream;

import ai.beestream.jitllm.tirex.model.TiRex2Config;

/**
 * The two recurrent time-mixer cells of the CPU reference. Both map {@code [rows · S · D]} to
 * {@code [rows · S · D]}, processing each row's sequence independently.
 */
final class Cells {

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
        float[] q = w.q().apply(x, n);
        float[] k = w.k().apply(x, n);
        float[] v = w.v().apply(x, n);
        float[] o = w.o().apply(x, n);
        float[] ig = w.igate().apply(x, n);
        float[] fg = w.fgate().apply(x, n);
        int dqk = w.q().out() / heads;
        int dv = w.v().out() / heads;
        double scale = 1.0 / Math.sqrt(dqk);
        float[] h = new float[n * w.v().out()];

        IntStream.range(0, rows * heads).parallel().forEach(rh -> {
            int r = rh / heads, hd = rh % heads;
            double[] cumF = new double[seq];
            double[] in = new double[seq];
            double acc = 0;
            for (int t = 0; t < seq; t++) {
                int gi = (r * seq + t) * heads + hd;
                acc += Ops.logSigmoid(Ops.softCap(fg[gi], TiRex2Config.GATE_SOFT_CAP));
                cumF[t] = acc;
                in[t] = Ops.softCap(ig[gi], TiRex2Config.GATE_SOFT_CAP);
            }
            double[] c = new double[seq];
            double[] num = new double[dv];
            for (int t = 0; t < seq; t++) {
                double m = cumF[t];
                for (int s = 0; s <= t; s++) {
                    m = Math.max(m, cumF[t] - cumF[s] + in[s]);
                }
                int qo = (r * seq + t) * heads * dqk + hd * dqk;
                double den = 0;
                java.util.Arrays.fill(num, 0);
                for (int s = 0; s <= t; s++) {
                    int ko = (r * seq + s) * heads * dqk + hd * dqk;
                    double dot = 0;
                    for (int j = 0; j < dqk; j++) {
                        dot += (double) q[qo + j] * k[ko + j];
                    }
                    double cts = dot * scale * Math.exp(cumF[t] - cumF[s] + in[s] - m);
                    den += cts;
                    int vo = (r * seq + s) * heads * dv + hd * dv;
                    for (int j = 0; j < dv; j++) {
                        num[j] += cts * v[vo + j];
                    }
                }
                double norm = Math.max(Math.abs(den), Math.exp(-m)) + TiRex2Config.MLSTM_EPS;
                int ho = (r * seq + t) * heads * dv + hd * dv;
                for (int j = 0; j < dv; j++) {
                    h[ho + j] = (float) (num[j] / norm);
                }
            }
        });

        float[] hn = Ops.headLayerNorm(h, n, heads, dv, w.headNorm(), TiRex2Config.NORM_EPS);
        for (int i = 0; i < hn.length; i++) {
            hn[i] = (float) (Ops.sigmoid(o[i]) * hn[i]);
        }
        return w.outProj().apply(hn, n);
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
        int dh = d / heads;
        int n = rows * seq;
        // wx[g][(r·seq + t)·d + h·dh + o] = Σ_j x[(r·seq + t)·d + h·dh + j] · W_g[h][o][j]
        float[][] wx = new float[4][n * d];
        IntStream.range(0, n).parallel().forEach(rt -> {
            int xo = rt * d;
            for (int g = 0; g < 4; g++) {
                float[] wg = w.gates()[g];
                for (int h = 0; h < heads; h++) {
                    for (int o = 0; o < dh; o++) {
                        int wo = (h * dh + o) * dh;
                        double acc = 0;
                        for (int j = 0; j < dh; j++) {
                            acc += (double) x[xo + h * dh + j] * wg[wo + j];
                        }
                        wx[g][xo + h * dh + o] = (float) acc;
                    }
                }
            }
        });

        float[] r = w.r();
        float[] b = w.b();
        float[] out = new float[n * d];
        IntStream.range(0, rows).parallel().forEach(row -> {
            double[] y = new double[d], c = new double[d], nn = new double[d], m = new double[d];
            double[] raw = new double[4 * d];
            for (int t = 0; t < seq; t++) {
                int base = (row * seq + t) * d;
                // raw[g·d + h·dh + e] = Wx + Σ_p y[h·dh + p]·R[h][p][g][e] + b[h][g][e]
                for (int h = 0; h < heads; h++) {
                    for (int g = 0; g < 4; g++) {
                        for (int e = 0; e < dh; e++) {
                            raw[g * d + h * dh + e] = wx[g][base + h * dh + e] + b[(h * 4 + g) * dh + e];
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
        return Ops.headLayerNorm(out, n, heads, dh, w.groupNorm(), TiRex2Config.NORM_EPS);
    }
}
