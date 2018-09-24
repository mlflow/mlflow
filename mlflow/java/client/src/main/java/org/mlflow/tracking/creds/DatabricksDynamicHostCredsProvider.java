package org.mlflow.tracking.creds;

import java.util.Map;

import com.google.common.annotations.VisibleForTesting;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DatabricksDynamicHostCredsProvider implements MlflowHostCredsProvider {
  private static final Logger logger = LoggerFactory.getLogger(
    DatabricksDynamicHostCredsProvider.class);

  private final Map<String, String> configProvider;

  private DatabricksDynamicHostCredsProvider(Map<String, String> configProvider) {
    this.configProvider = configProvider;
  }

  public static DatabricksDynamicHostCredsProvider createIfAvailable() {
    return createIfAvailable("com.databricks.config.DatabricksClientSettingsProvider");
  }

  @VisibleForTesting
  static DatabricksDynamicHostCredsProvider createIfAvailable(String className) {
    try {
      Class<?> cls = Class.forName(className);
      return new DatabricksDynamicHostCredsProvider((Map<String, String>) cls.newInstance());
    } catch (ClassNotFoundException e) {
      return null;
    } catch (IllegalAccessException | InstantiationException e) {
      logger.warn("Found but failed to invoke dynamic config provider", e);
      return null;
    }

  }

  @Override
  public MlflowHostCreds getHostCreds() {
    return new BasicMlflowHostCreds(
      configProvider.get("host"),
      configProvider.get("username"),
      configProvider.get("password"),
      configProvider.get("token"),
      "true".equals(configProvider.get("shouldIgnoreTlsVerification"))
    );
  }

  @Override
  public void refresh() {
    // no-op
  }
}
