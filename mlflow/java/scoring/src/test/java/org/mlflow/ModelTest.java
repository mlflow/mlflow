package org.mlflow.models;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import org.junit.Assert;
import org.junit.Test;
import org.mlflow.mleap.MLeapFlavor;

/**
 * Unit tests associated with MLflow model configuration parsing and other operations associated
 * with the {@link Model} class
 */
public class ModelTest {
  @Test
  public void testModelIsLoadedFromYamlUsingConfigPathCorrectly() {
    String configPath = getClass().getResource("sample_model_root/MLmodel").getFile();
    try {
      Model model = Model.fromConfigPath(configPath);
      Assert.assertTrue(model.getFlavor(MLeapFlavor.FLAVOR_NAME, MLeapFlavor.class).isPresent());
      Assert.assertTrue(model.getUtcTimeCreated().isPresent());
      Assert.assertTrue(model.getModelUuid().isPresent());
      Assert.assertTrue(model.getMlflowVersion().isPresent());
    } catch (IOException e) {
      e.printStackTrace();
      Assert.fail("Encountered an exception while reading the model from a configuration path!");
    }
  }

  @Test
  public void testModelIsLoadedCorrectlyWhenModelUuidDoesNotExist() {
    String configPath = getClass().getResource("sample_model_root/MLmodel.no.model_uuid").getFile();
    try {
      Model model = Model.fromConfigPath(configPath);
      Assert.assertFalse(model.getModelUuid().isPresent());
    } catch (IOException e) {
      e.printStackTrace();
      Assert.fail("Encountered an exception while reading the model from a configuration path!");
    }
  }

  @Test
  public void testModelIsLoadedCorrectlyWhenMlflowVersionDoesNotExist() {
    String configPath = getClass().getResource("sample_model_root/MLmodel.no.mlflow_version").getFile();
    try {
      Model model = Model.fromConfigPath(configPath);
      Assert.assertFalse(model.getMlflowVersion().isPresent());
    } catch (IOException e) {
      e.printStackTrace();
      Assert.fail("Encountered an exception while reading the model from a configuration path!");
    }
  }

  @Test
  public void testModelIsLoadedCorrectlyWhenDatabricksRuntimeExists() {
    String configPath = getClass().getResource("sample_model_root/MLmodel.with.databricks_runtime").getFile();
    try {
      Model model = Model.fromConfigPath(configPath);
      Assert.assertTrue(model.getDatabricksRuntime().isPresent());
    } catch (IOException e) {
      e.printStackTrace();
      Assert.fail("Encountered an exception while reading the model from a configuration path!");
    }
  }

  @Test
  public void testModelIsLoadedCorrectlyWhenMetadataExists() {
    String configPath = getClass().getResource("sample_model_root/MLmodel.with.metadata").getFile();
    try {
      Model model = Model.fromConfigPath(configPath);
      Assert.assertTrue(model.getMetadata().isPresent());
      JsonNode metadata = model.getMetadata().get();
      Assert.assertEquals("metadata_value", metadata.get("metadata_key").asText());
      Assert.assertTrue(metadata.get("stateful").asBoolean());
      Assert.assertEquals(0.978, metadata.get("accuracy").asDouble(), 1e-3);
      Assert.assertTrue(metadata.get("other").isNull());
    } catch (IOException e) {
      e.printStackTrace();
      Assert.fail("Encountered an exception while reading the model from a configuration path!");
    }
  }

  @Test
  public void testModelIsLoadedFromYamlUsingRootPathCorrectly() {
    String rootPath = getClass().getResource("sample_model_root").getFile();
    try {
      Model model = Model.fromRootPath(rootPath);
      Assert.assertTrue(model.getFlavor(MLeapFlavor.FLAVOR_NAME, MLeapFlavor.class).isPresent());
      Assert.assertTrue(model.getUtcTimeCreated().isPresent());
    } catch (IOException e) {
      e.printStackTrace();
      Assert.fail("Encountered an exception while reading the model from a configuration path!");
    }
  }
}
