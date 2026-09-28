package ai.beestream.jitllm.tirex.cpu;

/**
 * Plain-Java FP32 operators of the CPU reference. Activations are row-major {@code float[]} of
 * {@code rows × cols}; weights are PyTorch {@code [out, in]}. Reductions accumulate in double so the
 * reference is a stable oracle for the device kernels, not a speed target.
 */
public final class Ops {

    /** torch.finfo(float32).eps: what F.rms_norm uses when no eps is given (the attention qk-norm). */
    public static final float FLOAT32_EPS = Math.ulp(1.0f);

    private Ops() {}

    /** {@code y[r] = W·x[r] + b}; {@code w} is {@code [out, in]}, {@code b} may be null. */
    public static float[] linear(float[] x, int rows, float[] w, float[] b, int in, int out) {
        float[] y = new float[rows * out];
        for (int r = 0; r < rows; r++) {
            int xo = r * in;
            for (int o = 0; o < out; o++) {
                int wo = o * in;
                double acc = 0;
                for (int i = 0; i < in; i++) {
                    acc += (double) x[xo + i] * w[wo + i];
                }
                y[r * out + o] = (float) (b == null ? acc : acc + b[o]);
            }
        }
        return y;
    }

    /** {@code x · rsqrt(mean(x²) + eps) · w} over each row; {@code w} may be null. */
    public static float[] rmsNorm(float[] x, int rows, int cols, float[] w, float eps) {
        float[] y = new float[x.length];
        for (int r = 0; r < rows; r++) {
            int o = r * cols;
            double ss = 0;
            for (int c = 0; c < cols; c++) {
                ss += (double) x[o + c] * x[o + c];
            }
            float inv = (float) (1.0 / Math.sqrt(ss / cols + eps));
            for (int c = 0; c < cols; c++) {
                y[o + c] = w == null ? x[o + c] * inv : x[o + c] * inv * w[c];
            }
        }
        return y;
    }

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

    public static float silu(float v) {
        return (float) (v / (1.0 + Math.exp(-v)));
    }

    public static float sigmoid(float v) {
        return (float) (1.0 / (1.0 + Math.exp(-v)));
    }

    public static void siluInPlace(float[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = silu(x[i]);
        }
    }

    public static void addInPlace(float[] acc, float[] x) {
        for (int i = 0; i < acc.length; i++) {
            acc[i] += x[i];
        }
    }

    /** {@code output(SiLU(hidden(x))) + residual(x)}, the TiRex ResidualBlock without layer norm. */
    public static float[] residualBlock(float[] x, int rows, CpuWeights.ResidualBlock w) {
        float[] h = linear(x, rows, w.hiddenW(), w.hiddenB(), w.in(), w.hidden());
        siluInPlace(h);
        float[] y = linear(h, rows, w.outputW(), w.outputB(), w.hidden(), w.out());
        addInPlace(y, linear(x, rows, w.residualW(), w.residualB(), w.in(), w.out()));
        return y;
    }
}
