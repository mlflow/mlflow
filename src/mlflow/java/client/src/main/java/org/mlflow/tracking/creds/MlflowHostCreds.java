package org.mlflow.tracking.creds;

/**
 * Provides a hostname and optional authentication for talking to an MLflow server.
 */
public interface MlflowHostCreds {
  /** Hostname (e.g., http://localhost:5000) to MLflow server. */
  String getHost();

  /**
   * Username to use with Basic authentication when talking to server.
   * If this is specified, password must also be specified.
   */
  String getUsername();

  /**
   * Password to use with Basic authentication when talking to server.
   * If this is specified, username must also be specified.
   */
  String getPassword();

  /**
   * Token to use with Bearer authentication when talking to server.
   * If provided, user/password authentication will be ignored.
   */
  String getToken();

  /**
   * If true, we will not verify the server's hostname or TLS certificate.
   * This is useful for certain testing situations, but should never be true in production.
   */
  boolean shouldIgnoreTlsVerification();
}
