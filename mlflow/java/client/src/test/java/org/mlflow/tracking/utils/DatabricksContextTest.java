package org.mlflow.tracking.utils;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import org.testng.Assert;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.util.*;

public class DatabricksContextTest {
  private static Map<String, String> baseMap = new HashMap<>();

  public static class MyDynamicProvider extends AbstractMap<String, String> {
    @Override
    public Set<Entry<String, String>> entrySet() {
      return baseMap.entrySet();
    }
  }

  @BeforeMethod
  public static void beforeMethod() {
    baseMap = new HashMap<>();
  }


  @Test
  public void testIsInDatabricksNotebook() {
    baseMap.put("notebookId", "1");
    DatabricksContext context = DatabricksContext.createIfAvailable(MyDynamicProvider.class.getName());
    Assert.assertTrue(context.isInDatabricksNotebook());
  }

  @Test
  public void testGetNotebookId() {
    baseMap.put("notebookId", "1");
    DatabricksContext context = DatabricksContext.createIfAvailable(MyDynamicProvider.class.getName());
    Assert.assertEquals(context.getNotebookId(), "1");
  }

  @Test
  public void testGetTags() {
    // Will return empty map if not in Databricks notebook.
    {
      baseMap.put("notebookId", null);
      baseMap.put("notebookPath", null);
      DatabricksContext context = DatabricksContext.createIfAvailable(MyDynamicProvider.class.getName());
      Assert.assertFalse(context.isInDatabricksNotebook());
      Assert.assertEquals(context.getTags(), Maps.newHashMap());
    }

    // Will return all tags if context is set as expected.
    {
      baseMap = new HashMap<>();
      Map<String, String> expectedTags = ImmutableMap.of(
        MlflowTagConstants.DATABRICKS_NOTEBOOK_ID, "1",
        MlflowTagConstants.DATABRICKS_NOTEBOOK_PATH, "test-path",
        MlflowTagConstants.SOURCE_TYPE, "NOTEBOOK",
        MlflowTagConstants.SOURCE_NAME, "test-path");
      baseMap.put("notebookId", "1");
      baseMap.put("notebookPath", "test-path");
      DatabricksContext context = DatabricksContext.createIfAvailable(MyDynamicProvider.class.getName());
      Assert.assertEquals(context.getTags(), expectedTags);
    }

    // Will not set notebook path tags if context doesn't have a notebookPath member.
    {
      baseMap = new HashMap<>();
      Map<String, String> expectedTags = ImmutableMap.of(
        MlflowTagConstants.DATABRICKS_NOTEBOOK_ID, "1");
      baseMap.put("notebookId", "1");
      baseMap.put("notebookPath", null);
      DatabricksContext context = DatabricksContext.createIfAvailable(MyDynamicProvider.class.getName());
      Assert.assertEquals(context.getTags(), expectedTags);
    }
  }
}
