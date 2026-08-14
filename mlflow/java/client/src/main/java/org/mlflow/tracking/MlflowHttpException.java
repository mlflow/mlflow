package org.mlflow.tracking;

/**
 * Returned when an HTTP API request to a remote Tracking service returns an error code.
 */
public class MlflowHttpException extends MlflowClientException {

  public MlflowHttpException(int statusCode, String reasonPhrase) {
    super("statusCode=" + statusCode + " reasonPhrase=[" + reasonPhrase +"]");
    this.statusCode = statusCode;
    this.reasonPhrase = reasonPhrase;
  }

  public MlflowHttpException(int statusCode, String reasonPhrase, String bodyMessage) {
    super("statusCode=" + statusCode + " reasonPhrase=[" + reasonPhrase + "] bodyMessage=["
      + bodyMessage + "]");
    this.statusCode = statusCode;
    this.reasonPhrase = reasonPhrase;
    this.bodyMessage = bodyMessage;
  }

  private int statusCode;

  public int getStatusCode() {
    return statusCode;
  }

  private String reasonPhrase;

  public String getReasonPhrase() {
    return reasonPhrase;
  }

  private String bodyMessage;

  public String getBodyMessage() {
    return bodyMessage;
  }
}
