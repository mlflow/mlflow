package com.databricks.mlflow.models;

import org.junit.Test;
import org.junit.Assert;

import com.databricks.mlflow.mleap.MLeapFlavor;

import java.io.File;
import java.io.IOException;

/**
 * Unit tests associated with MLFlow model configuration parsing and
 * other operations associated with the {@link Model} class
 */
public class ModelTests {
    @Test
    public void testModelIsLoadedFromYamlUsingConfigPathCorrectly() {
        String configPath = getClass().getResource("sample_model_root/MLModel").getFile();
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

    @Test
    public void testModelIsLoadedFromYamlUsingRootPathCorrectly() {
        String rootPath = getClass().getResource("sample_model_root").getFile();
        try {
            Model model = Model.fromRootPath(rootPath);
            Assert.assertTrue(
                model.getFlavor(MLeapFlavor.FLAVOR_NAME, MLeapFlavor.class).isPresent());
            Assert.assertTrue(model.getUtcTimeCreated().isPresent());
        } catch (IOException e) {
            Assert.fail(
                "Encountered an exception while reading the model from a configuration path!");
        }
    }
}
