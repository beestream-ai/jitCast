package ai.beestream.jitllm.tirex.cpu;

import java.util.Arrays;
import java.util.stream.IntStream;

import ai.beestream.jitllm.tirex.model.ForwardPass;
import ai.beestream.jitllm.tirex.model.ModelBatch;
import ai.beestream.jitllm.tirex.model.TiRex2Config;
import ai.beestream.jitllm.tirex.model.TiRex2Weights;

/**
 * The FP32 CPU reference of {@code TiRex2.forward()}: the oracle the TornadoVM kernels are tested
 * against, stage by stage, with the same stage names as the golden activations.
 *
 * <p>Activations between stages are {@code [rows · patches · D]}, row-major.
 */
public final class CpuReference implements ForwardPass {

    /** Receives each stage's output; names match {@code scripts/dump_golden.py}. */
    @FunctionalInterface
    public interface StageSink {
        StageSink NONE = (stage, values) -> {};

        void accept(String stage, float[] values);
    }

    private final TiRex2Config config;
    private final CpuWeights weights;

    public CpuReference(TiRex2Weights weights) {
        this.config = weights.config();
        this.weights = new CpuWeights(weights);
    }

    public TiRex2Config config() {
        return config;
    }

    /** {@code input_patch_embedding(cat(tokens, mask))}: {@code [rows · patches · D]}. */
    public float[] embed(InputStage.Result input) {
        return Ops.residualBlock(input.tokens(), input.rows() * input.patches(), weights.inputEmbedding);
    }

    @Override
    public float[][][] forward(ModelBatch batch) {
        return forward(batch, StageSink.NONE);
    }

    /** The whole forward pass: {@code [rows][quantiles][patches · patch]}, rescaled to data units. */
    public float[][][] forward(ModelBatch batch, StageSink sink) {
        int rows = batch.x().length;
        int seq = config.numPatches();
        int d = config.embeddingDim();
        int[] known = knownCovariates(batch);

        InputStage.Result input = InputStage.apply(batch.x(), config);
        float[] h = embed(input);
        sink.accept("input_patch_embedding", h);
        for (int i = 0; i < weights.blocks.size(); i++) {
            CpuWeights.Block block = weights.blocks.get(i);
            h = timeMixer(h, rows, block, known);
            sink.accept("stack." + i + ".time_mixer", h);
            h = variateMixer(h, rows, block, batch.group(), batch.target());
            sink.accept("stack." + i + ".variate_mixer", h);
        }
        h = Ops.rmsNorm(h, rows * seq, d, weights.stackOutNorm, config.stackOutNormEps());
        sink.accept("stack_out_norm", h);
        float[] y = Ops.residualBlock(h, rows * seq, weights.outputEmbedding);
        sink.accept("output_patch_embedding", y);
        return unpatch(y, rows, input.scaler());
    }

    /** Covariate rows with at least one observation in the last {@code futureLen} steps. */
    int[] knownCovariates(ModelBatch batch) {
        int len = config.paddedLength();
        return IntStream.range(0, batch.x().length).filter(r -> {
            if (batch.target()[r]) {
                return false;
            }
            for (int t = len - config.futureLen(); t < len; t++) {
                if (!Float.isNaN(batch.x()[r][t])) {
                    return true;
                }
            }
            return false;
        }).toArray();
    }

    /**
     * BiXLSTM: {@code y = cell(RMSNorm(x))}; for known covariates the reverse pass over the flipped
     * sequence is concatenated and projected back ({@code combination}); then two residuals, around
     * the cell and around the SwiGLU FFN.
     */
    private float[] timeMixer(float[] x, int rows, CpuWeights.Block w, int[] known) {
        int seq = config.numPatches();
        int d = config.embeddingDim();
        int n = rows * seq;
        float[] xn = Ops.rmsNorm(x, n, d, w.normLstm(), TiRex2Config.NORM_EPS);
        float[] y = cell(xn, rows, w.cell());
        if (known.length > 0) {
            float[] rev = new float[known.length * seq * d];
            for (int k = 0; k < known.length; k++) {
                for (int t = 0; t < seq; t++) {
                    System.arraycopy(xn, (known[k] * seq + (seq - 1 - t)) * d, rev, (k * seq + t) * d, d);
                }
            }
            float[] revOut = cell(rev, known.length, w.cell());
            float[] cat = new float[known.length * seq * 2 * d];
            for (int k = 0; k < known.length; k++) {
                for (int t = 0; t < seq; t++) {
                    int dst = (k * seq + t) * 2 * d;
                    System.arraycopy(y, (known[k] * seq + t) * d, cat, dst, d);
                    System.arraycopy(revOut, (k * seq + (seq - 1 - t)) * d, cat, dst + d, d);
                }
            }
            float[] combined = w.combination().apply(cat, known.length * seq);
            for (int k = 0; k < known.length; k++) {
                System.arraycopy(combined, k * seq * d, y, known[k] * seq * d, seq * d);
            }
        }
        Ops.addInPlace(y, x);
        float[] ffn = Ops.swiGlu(Ops.rmsNorm(y, n, d, w.normFfn(), TiRex2Config.NORM_EPS), n,
                w.upGate(), w.up(), w.down());
        Ops.addInPlace(ffn, y);
        return ffn;
    }

    private float[] cell(float[] x, int rows, CpuWeights.Cell cell) {
        int seq = config.numPatches();
        int d = config.embeddingDim();
        return switch (cell) {
            case CpuWeights.Mlstm m -> Cells.mlstm(x, rows, seq, d, config.timeMixerHeads(), m);
            case CpuWeights.Slstm s -> Cells.slstm(x, rows, seq, d, config.slstmHeads(), s);
        };
    }

    /**
     * Attention across the rows of each group, per patch, then a SiLU MLP; both pre-normalised and
     * residual. Rows alone in their group pass through unchanged, and a batch of only such rows
     * skips the block (the univariate case).
     */
    private float[] variateMixer(float[] x, int rows, CpuWeights.Block w, float[] group, boolean[] target) {
        int[] kept = IntStream.range(0, rows).filter(r -> {
            for (int o = 0; o < rows; o++) {
                if (o != r && group[o] == group[r]) {
                    return true;
                }
            }
            return false;
        }).toArray();
        if (kept.length == 0) {
            return x;
        }
        int seq = config.numPatches();
        int d = config.embeddingDim();
        int heads = config.variateHeads();
        int dh = d / heads;
        int nk = kept.length;
        int n = nk * seq;

        // gather the kept rows: [nk · seq · d]
        float[] xk = new float[n * d];
        for (int k = 0; k < nk; k++) {
            System.arraycopy(x, kept[k] * seq * d, xk, k * seq * d, seq * d);
        }
        float[] nrm = Ops.rmsNorm(xk, n, d, w.normAttn(), TiRex2Config.NORM_EPS);
        float[] q = Ops.rmsNorm(w.wq().apply(nrm, n), n * heads, dh, null, Ops.FLOAT32_EPS);
        float[] kk = Ops.rmsNorm(w.wk().apply(nrm, n), n * heads, dh, null, Ops.FLOAT32_EPS);
        float[] v = w.wv().apply(nrm, n);
        double scale = 1.0 / Math.sqrt(dh);
        float[] attn = new float[n * d];

        IntStream.range(0, seq * heads).parallel().forEach(th -> {
            int t = th / heads, h = th % heads;
            double[] s = new double[nk];
            for (int qi = 0; qi < nk; qi++) {
                int qo = (qi * seq + t) * d + h * dh;
                double max = Double.NEGATIVE_INFINITY;
                for (int ki = 0; ki < nk; ki++) {
                    boolean allowed = group[kept[qi]] == group[kept[ki]] && !(!target[kept[qi]] && target[kept[ki]]);
                    if (!allowed) {
                        s[ki] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    int ko = (ki * seq + t) * d + h * dh;
                    double dot = 0;
                    for (int j = 0; j < dh; j++) {
                        dot += (double) q[qo + j] * kk[ko + j];
                    }
                    s[ki] = dot * scale;
                    max = Math.max(max, s[ki]);
                }
                double sum = 0;
                for (int ki = 0; ki < nk; ki++) {
                    s[ki] = s[ki] == Double.NEGATIVE_INFINITY ? 0 : Math.exp(s[ki] - max);
                    sum += s[ki];
                }
                for (int j = 0; j < dh; j++) {
                    double acc = 0;
                    for (int ki = 0; ki < nk; ki++) {
                        acc += s[ki] * v[(ki * seq + t) * d + h * dh + j];
                    }
                    attn[qo + j] = (float) (acc / sum);
                }
            }
        });
        float[] out = w.wo().apply(attn, n);
        Ops.addInPlace(out, xk);
        float[] hidden = w.ffnIn().apply(Ops.rmsNorm(out, n, d, w.normFfnAttn(), TiRex2Config.NORM_EPS), n);
        Ops.siluInPlace(hidden);
        float[] mlp = w.ffnOut().apply(hidden, n);
        Ops.addInPlace(mlp, out);

        float[] result = Arrays.copyOf(x, x.length);
        for (int k = 0; k < nk; k++) {
            System.arraycopy(mlp, k * seq * d, result, kept[k] * seq * d, seq * d);
        }
        return result;
    }

    /**
     * {@code [rows · patches · (Q·P)]} → {@code [rows][Q][patches·P]}, then undo the scaler:
     * {@code sinh(clamp(y, ±20))·scale + loc}, with no sinh on binary rows.
     */
    private float[][][] unpatch(float[] y, int rows, InputStage.ScalerState s) {
        int seq = config.numPatches();
        int p = config.patchSize();
        int nq = config.numQuantiles();
        float[][][] out = new float[rows][nq][seq * p];
        for (int r = 0; r < rows; r++) {
            boolean sinh = config.useArcsinh() && !s.binary()[r];
            for (int t = 0; t < seq; t++) {
                int base = (r * seq + t) * nq * p;
                for (int q = 0; q < nq; q++) {
                    for (int i = 0; i < p; i++) {
                        float v = y[base + q * p + i];
                        if (sinh) {
                            float c = Math.max(-TiRex2Config.SINH_CLAMP, Math.min(TiRex2Config.SINH_CLAMP, v));
                            v = (float) Math.sinh(c);
                        }
                        out[r][q][t * p + i] = v * s.scale()[r] + s.loc()[r];
                    }
                }
            }
        }
        return out;
    }
}
