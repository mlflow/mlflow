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
    Map<String, String> tags = new HashMap<>();
    if (!isInDatabricksNotebook()) {
      return tags;
    }
    String notebookId = getNotebookId();
    if (notebookId != null) {
      tags.put(MlflowTagConstants.DATABRICKS_NOTEBOOK_ID, notebookId);
    }
    String notebookPath = getNotebookPath();
    if (notebookPath != null) {
      tags.put(MlflowTagConstants.SOURCE_NAME, notebookPath);
      tags.put(MlflowTagConstants.DATABRICKS_NOTEBOOK_PATH, notebookPath);
      tags.put(MlflowTagConstants.SOURCE_TYPE, "NOTEBOOK");
    }
    String webappUrl = getWebappUrl();
    if (webappUrl != null) {
      tags.put(MlflowTagConstants.DATABRICKS_WEBAPP_URL, webappUrl);
    }
    return tags;
  }

  public boolean isInDatabricksNotebook() {
    return configProvider.get("notebookId") != null;
  }

  /**
   * Should only be called if isInDatabricksNotebook() is true.
   */
  public String getNotebookId() {
    return configProvider.get("notebookId");
  }

  /**
   * Should only be called if isInDatabricksNotebook() is true.
   */
  private String getNotebookPath() {
    if (!isInDatabricksNotebook()) {
      throw new IllegalArgumentException(
        "getNotebookPath() should not be called when isInDatabricksNotebook() is false"
      );
    };
    return configProvider.get("notebookPath");
  }

  /**
   * Should only be called if isInDatabricksNotebook() is true.
   */
  private String getWebappUrl() {
    if (!isInDatabricksNotebook()) {
      throw new IllegalArgumentException(
        "getWebappUrl() should not be called when isInDatabricksNotebook() is false"
      );
    };
    return configProvider.get("host");
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
