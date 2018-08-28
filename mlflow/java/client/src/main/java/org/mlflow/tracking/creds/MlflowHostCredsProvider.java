package org.mlflow.tracking.creds;

/** Provides a dynamic, refreshable set of MlflowHostCreds. */
public interface MlflowHostCredsProvider {

  /** Returns a valid MlflowHostCreds. This may be cached. */
  MlflowHostCreds getHostCreds();

  /** Refreshes the underlying credentials. May be a no-op. */
  void refresh();
}
