package org.mlflow.sagemaker;

import ml.combust.mleap.runtime.frame.Transformer;

/** A {@link org.mlflow.sagemaker.Predictor} implementation for the MLeap model flavor */
public class MLeapPredictor extends Predictor {
  public MLeapPredictor(String modelDataPath, String inputSchemaPath) {}

  @Override
  protected PredictorDataWrapper predict(PredictorDataWrapper input) {
    throw new UnsupportedOperationException("Not yet implemented!");
  }

  /** @return The underlying MLeap pipeline transformer that this predictor uses for inference */
  public Transformer getPipeline() {
    throw new UnsupportedOperationException("Not yet implemented!");
  }
}
