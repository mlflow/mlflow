package org.mlflow.tracking.creds;

import java.util.Map;

import com.google.common.annotations.VisibleForTesting;
import org.mlflow.tracking.utils.DatabricksContext;
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
    Map<String, String> configProvider =
      DatabricksContext.getConfigProviderIfAvailable(DatabricksContext.CONFIG_PROVIDER_CLASS_NAME);
    if (configProvider == null) {
      return null;
    }
    return new DatabricksDynamicHostCredsProvider(configProvider);
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
