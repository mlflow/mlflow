package org.mlflow;

import org.junit.Assert;
import org.junit.Test;
import org.mlflow.mleap.MLeapLoader;
import org.mlflow.sagemaker.Predictor;
import org.mlflow.sagemaker.PredictorLoadingException;

/** Unit tests for deserializing MLflow models as generic {@link Predictor} objects for inference */
public class LoaderModuleTest {
  @Test
  public void testMLeapLoaderModuleDeserializesValidMLeapModelAsPredictor() {
    String modelPath = MLflowRootResourceProvider.getResourcePath("mleap_model");
    try {
      Predictor predictor = new MLeapLoader().load(modelPath);
    } catch (PredictorLoadingException e) {
      e.printStackTrace();
      Assert.fail("Encountered unexpected `PredictorLoadingException`!");
    }
  }
}
