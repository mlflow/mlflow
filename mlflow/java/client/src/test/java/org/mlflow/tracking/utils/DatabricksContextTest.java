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
  public void testGetTagsWithEmptyNotebookAndJobDetails() {
    // Will return empty map if not in Databricks notebook.
    {
      baseMap.put("notebookId", null);
      baseMap.put("notebookPath", null);
      DatabricksContext context = DatabricksContext.createIfAvailable(MyDynamicProvider.class.getName());
      Assert.assertFalse(context.isInDatabricksNotebook());
      Assert.assertEquals(context.getTags(), Maps.newHashMap());
    }
    // Will return empty map if not in Databricks job.
    {
      baseMap.put("jobId", null);
      baseMap.put("jobRunId", null);
      DatabricksContext context = DatabricksContext.createIfAvailable(MyDynamicProvider.class.getName());
      Assert.assertFalse(context.isInDatabricksNotebook());
      Assert.assertEquals(context.getTags(), Maps.newHashMap());
    }
    // Will return empty map if not config map is empty.
    {
      DatabricksContext context = DatabricksContext.createIfAvailable(MyDynamicProvider.class.getName());
      Assert.assertFalse(context.isInDatabricksNotebook());
      Assert.assertEquals(context.getTags(), Maps.newHashMap());
    }
  }

  @Test
  public void testGetTagsForNotebook() {
    // Will return all tags if notebook context is set as expected.
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

  @Test
  public void testGetTagsForJob() {
    // Will return all tags if job context is set as expected.
    {
      baseMap = new HashMap<>();
      Map<String, String> expectedTags = ImmutableMap.of(
        MlflowTagConstants.DATABRICKS_JOB_ID, "70",
        MlflowTagConstants.DATABRICKS_JOB_RUN_ID, "5",
        MlflowTagConstants.DATABRICKS_JOB_TYPE, "notebook",
        MlflowTagConstants.SOURCE_TYPE, "JOB",
        MlflowTagConstants.SOURCE_NAME, "jobs/70/run/5");
      baseMap.put("jobId", "70");
      baseMap.put("jobRunId", "5");
      baseMap.put("jobType", "notebook");
      DatabricksContext context = DatabricksContext.createIfAvailable(MyDynamicProvider.class.getName());
      Assert.assertEquals(context.getTags(), expectedTags);
    }
    // Will not set job type tag if job type is absent.
    {
      baseMap = new HashMap<>();
      Map<String, String> expectedTags = ImmutableMap.of(
        MlflowTagConstants.DATABRICKS_JOB_ID, "70",
        MlflowTagConstants.DATABRICKS_JOB_RUN_ID, "5",
        MlflowTagConstants.SOURCE_TYPE, "JOB",
        MlflowTagConstants.SOURCE_NAME, "jobs/70/run/5");
      baseMap.put("jobId", "70");
      baseMap.put("jobRunId", "5");
      DatabricksContext context = DatabricksContext.createIfAvailable(MyDynamicProvider.class.getName());
      Assert.assertEquals(context.getTags(), expectedTags);
    }
  }
}
