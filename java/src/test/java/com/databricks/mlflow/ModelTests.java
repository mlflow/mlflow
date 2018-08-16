package com.databricks.mlflow.models;

import org.junit.Test;
import org.junit.Assert;

import com.databricks.mlflow.mleap.MLeapFlavor;

import java.io.File;
import java.io.IOException;

/**
 * Unit tests for deserializing MLFlow models as generic
 * {@link com.databricks.mlflow.sagemaker.Predictor} objects for inference
 */
public class ModelTests {
    @Test
    public void testModelIsLoadedFromYamlCorrectly() {
        String configPath = getClass().getResource("MLModel").getFile();
        try {
            Model model = Model.fromConfigPath(configPath);
            Assert.assertTrue(
                model.getFlavor(MLeapFlavor.FLAVOR_NAME, MLeapFlavor.class).isPresent());
            Assert.assertTrue(model.getUtcTimeCreated().isPresent());
        } catch (IOException e) {
            Assert.fail(
                "Encountered an exception while reading the model from a configuration path!");
        }
    }
}
