package ai.beestream.jitllm.tirex.host;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.beestream.jitllm.tirex.api.Forecast;
import ai.beestream.jitllm.tirex.api.TimeSeries;
import ai.beestream.jitllm.tirex.model.ForwardPass;
import ai.beestream.jitllm.tirex.model.ModelBatch;
import ai.beestream.jitllm.tirex.model.TiRex2Config;

/**
 * {@code ForecastModel.forecast()} around any {@link ForwardPass}: windows of at most
 * {@code batchSize} series, batch assembly ({@code PostProcessor.transform_input} and the padding
 * of {@code TiRex2._predict}), the forward pass, slicing out the horizon, calibration
 * ({@code transform_output}) and the optional sign-flip test-time augmentation.
 *
 * <p>Contexts are always padded to the full model length ({@code pad_context=True}, the default).
 */
public final class ForecastPipeline {

    /** The default {@code batch_size} of {@code ForecastModel.forecast}. */
    public static final int DEFAULT_BATCH_SIZE = 512;

    private final TiRex2Config config;
    private final ForwardPass forward;
    private final int batchSize;
    private final boolean ttaDiff;
    private final boolean ttaSignFlip;

    public ForecastPipeline(TiRex2Config config, ForwardPass forward) {
        this(config, forward, DEFAULT_BATCH_SIZE, config.ttaDiff(), config.ttaSignFlip());
    }

    public ForecastPipeline(TiRex2Config config, ForwardPass forward, int batchSize, boolean ttaDiff, boolean ttaSignFlip) {
        if (batchSize < 1) {
            throw new IllegalArgumentException("batchSize must be >= 1");
        }
        this.config = config;
        this.forward = forward;
        this.batchSize = batchSize;
        this.ttaDiff = ttaDiff;
        this.ttaSignFlip = ttaSignFlip;
    }

    /** The same pipeline with sign-flip TTA on or off (the {@code tta_sign_flip} argument of predict). */
    public ForecastPipeline withSignFlip(boolean signFlip) {
        return new ForecastPipeline(config, forward, batchSize, ttaDiff, signFlip);
    }

    public List<Forecast> forecast(List<TimeSeries> series, int horizon) {
        if (horizon < 1) {
            throw new IllegalArgumentException("horizon must be >= 1, got " + horizon);
        }
        // like TiRex2._predict_once, a horizon beyond futureLen is truncated to it
        int h = Math.min(horizon, config.futureLen());
        List<Forecast> out = new ArrayList<>(series.size());
        for (int start = 0; start < series.size(); start += batchSize) {
            out.addAll(predict(series.subList(start, Math.min(series.size(), start + batchSize)), h));
        }
        return out;
    }

    /** {@code TiRex2.predict}: one pass, or the average of the pass and its sign-flipped mirror. */
    List<Forecast> predict(List<TimeSeries> series, int h) {
        List<float[][][]> a = predictOnce(series, h);
        if (!ttaSignFlip || series.isEmpty()) {
            return wrap(a);
        }
        List<float[][][]> b = predictOnce(series.stream().map(ForecastPipeline::negate).toList(), h);
        int[] complement = complement(config.quantiles());
        for (int s = 0; s < a.size(); s++) {
            float[][][] fa = a.get(s), fb = b.get(s);
            for (int t = 0; t < fa.length; t++) {
                for (int q = 0; q < fa[t].length; q++) {
                    for (int i = 0; i < h; i++) {
                        fa[t][q][i] = (fa[t][q][i] + -fb[t][complement[q]][i]) / 2;
                    }
                }
            }
        }
        return wrap(a);
    }

    private List<Forecast> wrap(List<float[][][]> values) {
        return values.stream().map(v -> new Forecast(config.quantiles().clone(), v)).toList();
    }

    /** The rows, group ids and target flags of one assembled batch, plus what the output side needs. */
    record Assembled(ModelBatch batch, int[][] targetRows, boolean[][] differenced, float[][] lastValues) {}

    Assembled assemble(List<TimeSeries> series, int h) {
        int n = series.size();
        int maxLen = series.stream().mapToInt(TimeSeries::length).max().orElse(0);
        int modelLen = config.paddedLength();
        List<float[]> rows = new ArrayList<>();
        List<Float> groups = new ArrayList<>();
        List<Boolean> targets = new ArrayList<>();
        int[][] targetRows = new int[n][];
        boolean[][] differenced = new boolean[n][];
        float[][] last = new float[n][];

        for (int s = 0; s < n; s++) {
            TimeSeries ts = series.get(s);
            int T = ts.length();
            int leftPad = maxLen - T;
            float group = s + 1;
            boolean[] diff = new boolean[ts.targets()];
            last[s] = new float[ts.targets()];
            boolean anyDiff = false;
            for (int v = 0; v < ts.targets(); v++) {
                diff[v] = ttaDiff && PostProcessor.isTrending(ts.target()[v]);
                anyDiff |= diff[v];
                last[s][v] = PostProcessor.lastValue(ts.target()[v]);
            }
            differenced[s] = diff;
            targetRows[s] = new int[ts.targets()];
            for (int v = 0; v < ts.targets(); v++) {
                float[] values = diff[v] && anyDiff ? PostProcessor.diff(ts.target()[v]) : ts.target()[v];
                targetRows[s][v] = rows.size();
                rows.add(place(values, T, leftPad, h, maxLen, modelLen));
                groups.add(group);
                targets.add(true);
            }
            for (int v = 0; v < ts.pastCount(); v++) {
                rows.add(place(ts.pastCovariates()[v], T, leftPad, h, maxLen, modelLen));
                groups.add(group);
                targets.add(false);
            }
            for (int v = 0; v < ts.futureCount(); v++) {
                float[] fc = ts.futureCovariates()[v];
                if (fc.length < T + h) {
                    throw new IllegalArgumentException("future covariates of series " + s + " need at least "
                            + (T + h) + " steps, got " + fc.length);
                }
                rows.add(place(fc, T + h, leftPad, 0, maxLen, modelLen));
                groups.add(group);
                targets.add(false);
            }
        }
        float[] g = new float[rows.size()];
        boolean[] t = new boolean[rows.size()];
        for (int i = 0; i < g.length; i++) {
            g[i] = groups.get(i);
            t[i] = targets.get(i);
        }
        return new Assembled(new ModelBatch(rows.toArray(float[][]::new), g, t), targetRows, differenced, last);
    }

    /**
     * Lays out the first {@code count} values of a row the way transform_input and _predict do:
     * left-padded to the batch's longest context, right-padded over the horizon ({@code rightPad})
     * and then to {@code futureLen}, then left-padded (or cut from the left) to the model length.
     */
    private float[] place(float[] values, int count, int leftPad, int rightPad, int maxLen, int modelLen) {
        // transform_input: [leftPad NaN][values][rightPad NaN] = maxLen + h; _predict adds futureLen - h
        int len = maxLen + config.futureLen();
        float[] row = new float[modelLen];
        Arrays.fill(row, Float.NaN);
        int offset = modelLen - len; // left pad (> 0) or cut from the left (< 0)
        for (int i = 0; i < count; i++) {
            int pos = offset + leftPad + i;
            if (pos >= 0) {
                row[pos] = values[i];
            }
        }
        return row;
    }

    /** One pass: assemble, forward, slice the horizon, calibrate each target row. */
    List<float[][][]> predictOnce(List<TimeSeries> series, int h) {
        Assembled a = assemble(series, h);
        float[][][] out = forward.forward(a.batch());
        // pred[..., :-P][..., -futureLen:][..., :h]
        int from = config.paddedLength() - config.patchSize() - config.futureLen();
        List<float[][][]> result = new ArrayList<>(series.size());
        for (int s = 0; s < series.size(); s++) {
            int vt = a.targetRows()[s].length;
            float[][][] f = new float[vt][][];
            for (int v = 0; v < vt; v++) {
                float[][] rowOut = out[a.targetRows()[s][v]];
                float[][] q = new float[rowOut.length][];
                for (int k = 0; k < q.length; k++) {
                    q[k] = Arrays.copyOfRange(rowOut[k], from, from + h);
                }
                PostProcessor.calibrate(q, a.differenced()[s][v], a.lastValues()[s][v]);
                f[v] = q;
            }
            result.add(f);
        }
        return result;
    }

    static TimeSeries negate(TimeSeries ts) {
        return new TimeSeries(neg(ts.target()), neg(ts.pastCovariates()), neg(ts.futureCovariates()));
    }

    private static float[][] neg(float[][] m) {
        if (m == null) {
            return null;
        }
        float[][] r = new float[m.length][];
        for (int i = 0; i < m.length; i++) {
            r[i] = new float[m[i].length];
            for (int j = 0; j < m[i].length; j++) {
                r[i][j] = -m[i][j];
            }
        }
        return r;
    }

    /** {@code complement[i]} = the level closest to {@code 1 − q[i]}; fails unless within 1e-6. */
    static int[] complement(float[] q) {
        int[] c = new int[q.length];
        for (int i = 0; i < q.length; i++) {
            int best = 0;
            for (int j = 1; j < q.length; j++) {
                if (Math.abs(q[j] - (1 - q[i])) < Math.abs(q[best] - (1 - q[i]))) {
                    best = j;
                }
            }
            if (Math.abs(q[best] - (1 - q[i])) > 1e-6) {
                throw new IllegalStateException("sign-flip TTA needs a symmetric quantile set");
            }
            c[i] = best;
        }
        return c;
    }
}
