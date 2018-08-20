package org.mlflow.sagemaker;

import org.mlflow.mleap.LeapFrameUtils;
import org.mlflow.mleap.LeapFrameSchema;

import ml.combust.mleap.runtime.MleapContext;
import ml.combust.mleap.runtime.frame.Transformer;
import ml.combust.mleap.runtime.javadsl.BundleBuilder;
import ml.combust.mleap.runtime.javadsl.ContextBuilder;

import java.io.File;

/** A {@link org.mlflow.sagemaker.Predictor} implementation for the MLeap model flavor */
public class MLeapPredictor extends Predictor {
  private final Transformer pipelineTransformer;

  public MLeapPredictor(String modelDataPath, String inputSchemaPath) {
    MleapContext mleapContext = new ContextBuilder().createMleapContext();
    BundleBuilder bundleBuilder = new BundleBuilder();
    this.pipelineTransformer = bundleBuilder.load(new File(modelDataPath), mleapContext).root();
  }

  @Override
  protected DataFrame predict(DataFrame input) {
    throw new UnsupportedOperationException("Not yet implemented!");
  }

  /** @return The underlying MLeap pipeline transformer that this predictor uses for inference */
  public Transformer getPipeline() {
    throw new UnsupportedOperationException("Not yet implemented!");
  }
}
