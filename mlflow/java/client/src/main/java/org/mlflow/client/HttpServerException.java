package org.mlflow.client;

/**
 * HTTP server exception.
 */
public class HttpServerException extends HttpException {

  public HttpServerException(int statusCode, String reasonPhrase) {
    super(statusCode, reasonPhrase);
  }

  public HttpServerException(int statusCode, String reasonPhrase, String bodyMessage) {
    super(statusCode, reasonPhrase, bodyMessage);
  }
}
