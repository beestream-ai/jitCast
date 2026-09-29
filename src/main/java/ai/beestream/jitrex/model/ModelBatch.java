package ai.beestream.jitrex.model;

/**
 * The batch {@code TiRex2.forward()} receives: every target and covariate row of every series,
 * padded to the model length.
 *
 * @param x {@code [rows][contextLen + futureLen]}, NaN where there is no observation
 * @param group group id of each row; rows attend across variates only within their group
 * @param target whether each row is a target (covariate rows cannot attend to targets)
 */
public record ModelBatch(float[][] x, float[] group, boolean[] target) {

    public int rows() {
        return x.length;
    }
}
