package com.databricks.mlflow.client;

/**
 * HTTP client exception.
 */
public class HttpClientException extends HttpException {

    public HttpClientException(int statusCode, String reasonPhrase) {
        super(statusCode, reasonPhrase) ;
    }

    public HttpClientException(int statusCode, String reasonPhrase, String bodyMessage) {
        super(statusCode, reasonPhrase, bodyMessage) ;
    }
}
