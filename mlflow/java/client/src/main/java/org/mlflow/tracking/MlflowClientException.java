package org.mlflow.tracking;

public class MlflowClientException extends RuntimeException {
  public MlflowClientException(String message) {
    super(message);
  }
  public MlflowClientException(String message, Throwable cause) {
    super(message, cause);
  }
  public MlflowClientException(Throwable cause) {
    super(cause);
  }
}
