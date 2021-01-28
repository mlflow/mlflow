package org.mlflow.tracking.creds;

/** Credentials to talk to a Databricks-hosted MLflow server. */
public final class DatabricksMlflowHostCreds extends BasicMlflowHostCreds {

  public DatabricksMlflowHostCreds(String host, String username, String password) {
    super(host, username, password);
  }

  public DatabricksMlflowHostCreds(String host, String token) {
    super(host, token);
  }

  public DatabricksMlflowHostCreds(
      String host,
      String username,
      String password,
      String token,
      boolean shouldIgnoreTlsVerification) {
    super(host, username, password, token, shouldIgnoreTlsVerification);
  }
}
