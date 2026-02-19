package org.mlflow.tracking.creds;

abstract class DatabricksHostCredsProvider implements MlflowHostCredsProvider {

  @Override
  public abstract DatabricksMlflowHostCreds getHostCreds();

  @Override
  public abstract void refresh();

}
