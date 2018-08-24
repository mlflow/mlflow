package org.mlflow.client;

/**
 * HTTP exception.
 */
public class HttpException extends MlflowClientException {

  public HttpException(int statusCode, String reasonPhrase) {
    super("statusCode=" + statusCode + " reasonPhrase=" + reasonPhrase);
    this.statusCode = statusCode;
    this.reasonPhrase = reasonPhrase;
  }

  public HttpException(int statusCode, String reasonPhrase, String bodyMessage) {
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

  public void setStatusCode(int statusCode) {
    this.statusCode = statusCode;
  }

  private String reasonPhrase;

  public String getReasonPhrase() {
    return reasonPhrase;
  }

  public void setReasonPhrase(String reasonPhrase) {
    this.reasonPhrase = reasonPhrase;
  }

  private String bodyMessage;

  public String getBodyMessage() {
    return bodyMessage;
  }

  public void setBodyMessage(String bodyMessage) {
    this.bodyMessage = bodyMessage;
  }
}
