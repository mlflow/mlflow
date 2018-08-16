package com.databricks.mlflow;

import org.junit.Test;
import org.junit.Assert;

import com.databricks.mlflow.mleap.MLeapLoader;
import com.databricks.mlflow.sagemaker.Predictor;
import com.databricks.mlflow.sagemaker.PredictorLoadingException;

/**
 * Unit tests for deserializing MLFlow models as generic
 * {@link com.databricks.mlflow.sagemaker.Predictor} objects for inference
 */
public class LoaderModuleTests {
    @Test
    public void testMLeapLoaderModuleDeserializesValidMLeapModelAsPredictor() {
        String modelPath = getClass().getResource("mleap_model").getFile();
        try {
            Predictor predictor = (new MLeapLoader()).load(modelPath);
        } catch (PredictorLoadingException e) {
            e.printStackTrace();
            Assert.fail("Encountered unexpected `PredictorLoadingException`!");
        }
    }
}
