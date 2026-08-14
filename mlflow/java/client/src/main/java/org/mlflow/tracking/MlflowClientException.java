package org.mlflow.tracking;

/** Superclass of all exceptions thrown by the MlflowClient API. */
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
