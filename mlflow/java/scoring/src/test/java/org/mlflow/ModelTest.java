package org.mlflow.models;

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
