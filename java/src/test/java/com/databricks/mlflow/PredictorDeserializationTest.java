package com.databricks.mlflow.sagemaker;

import org.junit.Test;
import org.junit.Assert;

import com.databricks.mlflow.PredictorLoadingException;

import com.databricks.mlflow.mleap.MLeapLoader;

/**
 * Unit tests for deserializing MLFlow models as generic Predictor objects for inference
 */
public class PredictorDeserializationTest {
    @Test
    public void testMLeapLoaderModuleDeserializesValidMLeapModelAsPredictor() {
        try {
            String modelPath =
                "/Users/czumar/mlflow/java/src/test/artifacts/serialized_mleap_model";
            Predictor predictor = (new MLeapLoader()).load(modelPath);
        } catch (PredictorLoadingException e) {
            e.printStackTrace();
            Assert.fail("Encountered unexpected `PredictorLoadingException!`");
        }
    }
}
