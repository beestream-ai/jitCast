package ai.beestream.jitrex.model;

/** One engine's {@code TiRex2.forward()}: the rescaled quantile outputs over the whole padded length. */
@FunctionalInterface
public interface ForwardPass {

    /** @return {@code [rows][quantiles][contextLen + futureLen]} */
    float[][][] forward(ModelBatch batch);
}
