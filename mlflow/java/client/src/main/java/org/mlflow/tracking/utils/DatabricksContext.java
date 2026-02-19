package org.mlflow.tracking.utils;

import com.google.common.annotations.VisibleForTesting;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

public class DatabricksContext {
  public static final String CONFIG_PROVIDER_CLASS_NAME =
    "com.databricks.config.DatabricksClientSettingsProvider";
  private static final Logger logger = LoggerFactory.getLogger(
    DatabricksContext.class);
  private final Map<String, String> configProvider;

  private DatabricksContext(Map<String, String> configProvider) {
    this.configProvider = configProvider;
  }

  public static DatabricksContext createIfAvailable() {
    return createIfAvailable(CONFIG_PROVIDER_CLASS_NAME);
  }

  @VisibleForTesting
  static DatabricksContext createIfAvailable(String className) {
    Map<String, String> configProvider = getConfigProviderIfAvailable(className);
    if (configProvider == null) {
      return null;
    }
    return new DatabricksContext(configProvider);
  }

  public Map<String, String> getTags() {
    if (isInDatabricksNotebook()) {
      return getTagsForDatabricksNotebook();
    } else if (isInDatabricksJob()) {
      return getTagsForDatabricksJob();
    } else {
      return new HashMap<>();
    }
  }

  public boolean isInDatabricksNotebook() {
    return configProvider.get("notebookId") != null;
  }

  /**
   * Should only be called if isInDatabricksNotebook() is true.
   */
  private Map<String, String> getTagsForDatabricksNotebook() {
    Map<String, String> tagsForNotebook = new HashMap<>();
    String notebookId = getNotebookId();
    if (notebookId != null) {
      tagsForNotebook.put(MlflowTagConstants.DATABRICKS_NOTEBOOK_ID, notebookId);
    }
    String notebookPath = configProvider.get("notebookPath");
    if (notebookPath != null) {
      tagsForNotebook.put(MlflowTagConstants.SOURCE_NAME, notebookPath);
      tagsForNotebook.put(MlflowTagConstants.DATABRICKS_NOTEBOOK_PATH, notebookPath);
      tagsForNotebook.put(MlflowTagConstants.SOURCE_TYPE, "NOTEBOOK");
    }
    String webappUrl = configProvider.get("host");
    if (webappUrl != null) {
      tagsForNotebook.put(MlflowTagConstants.DATABRICKS_WEBAPP_URL, webappUrl);
    }
    return tagsForNotebook;
  }

  /**
   * Should only be called if isInDatabricksNotebook() is true.
   */
  public String getNotebookId() {
    if (!isInDatabricksNotebook()) {
      throw new IllegalArgumentException(
        "getNotebookId() should not be called when isInDatabricksNotebook() is false"
      );
    }
    return configProvider.get("notebookId");
  }

  public String getNotebookPath() {
    if (!isInDatabricksNotebook()) {
      throw new IllegalArgumentException(
        "getNotebookPath() should not be called when isInDatabricksNotebook() is false"
      );
    }
    return configProvider.get("notebookPath");
  }

  private boolean isInDatabricksJob() {
    return configProvider.get("jobId") != null;
  }

  /**
   * Should only be called if isInDatabricksJob() is true.
   */
  private Map<String, String> getTagsForDatabricksJob() {
    Map<String, String> tagsForJob = new HashMap<>();
    String jobId = configProvider.get("jobId");
    String jobRunId = configProvider.get("jobRunId");
    String jobType = configProvider.get("jobType");
    String webappUrl = configProvider.get("host");
    if (jobId != null && jobRunId != null) {
      tagsForJob.put(MlflowTagConstants.DATABRICKS_JOB_ID, jobId);
      tagsForJob.put(MlflowTagConstants.DATABRICKS_JOB_RUN_ID, jobRunId);
      tagsForJob.put(MlflowTagConstants.SOURCE_TYPE, "JOB");
      tagsForJob.put(MlflowTagConstants.SOURCE_NAME,
                          String.format("jobs/%s/run/%s", jobId, jobRunId));
    }
    if (jobType != null) {
      tagsForJob.put(MlflowTagConstants.DATABRICKS_JOB_TYPE, jobType);
    }
    if (webappUrl != null) {
      tagsForJob.put(MlflowTagConstants.DATABRICKS_WEBAPP_URL, webappUrl);
    }
    return tagsForJob;
  }

  public static Map<String, String> getConfigProviderIfAvailable(String className) {
    try {
      Class<?> cls = Class.forName(className);
      return (Map<String, String>) cls.newInstance();
    } catch (ClassNotFoundException e) {
      return null;
    } catch (IllegalAccessException | InstantiationException e) {
      logger.warn("Found but failed to invoke dynamic config provider", e);
      return null;
    }
  }
}
