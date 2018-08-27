package org.mlflow.tracking;

/**
 * HTTP server exception.
 */
public class MlflowHttpServerException extends MlflowHttpException {

  public MlflowHttpServerException(int statusCode, String reasonPhrase) {
    super(statusCode, reasonPhrase);
  }

  public MlflowHttpServerException(int statusCode, String reasonPhrase, String bodyMessage) {
    super(statusCode, reasonPhrase, bodyMessage);
  }
}
