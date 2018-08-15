package com.databricks.mlflow;

import org.junit.Test;
import org.junit.Assert;

import com.databricks.mlflow.mleap.MLeapLoader;

import ml.combust.mleap.runtime.frame.Transformer;

/**
 * Unit tests for deserializing MLFlow models with the MLeap flavor
 * as native MLeap Transformer objects
 */
public class PipelineDeserializationTest {
    @Test
    public void testMLeapLoaderModuleDeserializesValidMLeapModelAsTransformer() {
        try {
            String modelPath =
                "/Users/czumar/mlflow/java/src/test/artifacts/serialized_mleap_model";
            Transformer transformer = (new MLeapLoader()).loadPipeline(modelPath);
        } catch (PredictorLoadingException e) {
            e.printStackTrace();
            Assert.fail("Encountered unexpected `PredictorLoadingException!`");
        }
    }
}
