package org.mlflow.tracking;

/**
 * HTTP client exception.
 */
public class MlflowHttpClientException extends MlflowHttpException {

  public MlflowHttpClientException(int statusCode, String reasonPhrase) {
    super(statusCode, reasonPhrase);
  }

  public MlflowHttpClientException(int statusCode, String reasonPhrase, String bodyMessage) {
    super(statusCode, reasonPhrase, bodyMessage);
  }
}
