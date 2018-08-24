package org.mlflow.client;

import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpRequestBase;
import org.apache.http.conn.ssl.NoopHostnameVerifier;
import org.apache.http.conn.ssl.SSLConnectionSocketFactory;
import org.apache.http.conn.ssl.TrustSelfSignedStrategy;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.util.EntityUtils;
import org.apache.log4j.Logger;

import org.mlflow.client.creds.MlflowHostCreds;
import org.mlflow.client.creds.MlflowHostCredsProvider;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;


class HttpCaller {
  private static final Logger logger = Logger.getLogger(HttpCaller.class);
  private static final String BASE_API_PATH = "api/2.0/preview/mlflow";
  private HttpClient httpClient;
  private final MlflowHostCredsProvider hostCredsProvider;

  public HttpCaller(MlflowHostCredsProvider hostCredsProvider) {
    this.hostCredsProvider = hostCredsProvider;
  }

  String get(String path) throws Exception {
    logger.debug("Sending GET " + path);
    HttpGet request = new HttpGet();
    fillRequestSettings(request, path);
    HttpResponse response = httpClient.execute(request);
    checkError(response);
    String responseJosn = EntityUtils.toString(response.getEntity(), StandardCharsets.UTF_8);
    logger.debug("Response: " + responseJosn);
    return responseJosn;
  }

  // TODO(aaron) Convert to InputStream.
  byte[] getAsBytes(String path) throws Exception {
    logger.debug("Sending GET " + path);
    HttpGet request = new HttpGet();
    fillRequestSettings(request, path);
    HttpResponse response = httpClient.execute(request);
    checkError(response);
    byte[] bytes = EntityUtils.toByteArray(response.getEntity());
    logger.debug("response: #bytes=" + bytes.length);
    return bytes;
  }

  String post(String path, String json) throws Exception {
    logger.debug("Sending POST " + path + ": " + json);
    HttpPost request = new HttpPost();
    fillRequestSettings(request, path);
    request.setEntity(new StringEntity(json));
    HttpResponse response = httpClient.execute(request);
    checkError(response);
    String responseJson = EntityUtils.toString(response.getEntity(), StandardCharsets.UTF_8);
    logger.debug("Response: " + responseJson);
    return responseJson;
  }

  private void checkError(HttpResponse response) throws Exception {
    int statusCode = response.getStatusLine().getStatusCode();
    String reasonPhrase = response.getStatusLine().getReasonPhrase();
    if (isError(statusCode)) {
      String bodyMessage = EntityUtils.toString(response.getEntity());
      if (statusCode >= 400 && statusCode <= 499) {
        throw new HttpClientException(statusCode, reasonPhrase, bodyMessage);
      }
      if (statusCode >= 500 && statusCode <= 599) {
        throw new HttpServerException(statusCode, reasonPhrase, bodyMessage);
      }
      throw new HttpException(statusCode, reasonPhrase, bodyMessage);
    }
  }

  private void fillRequestSettings(HttpRequestBase request, String path) {
    MlflowHostCreds hostCreds = hostCredsProvider.getHostCreds();
    createHttpClientIfNecessary(hostCreds.getNoTlsVerify());
    String uri = hostCreds.getHost() + "/" + BASE_API_PATH + "/" + path;
    request.setURI(URI.create(uri));
    String username = hostCreds.getUsername();
    String password = hostCreds.getPassword();
    String token = hostCreds.getToken();
    if (username != null && password != null) {
      String authHeader = Base64.getEncoder()
        .encodeToString((username + ":" + password).getBytes(StandardCharsets.UTF_8));
      request.addHeader("Authorization", "Basic " + authHeader);
    } else if (token != null) {
      request.addHeader("Authorization", "Bearer " + token);
    }
  }

  private boolean isError(int statusCode) {
    return statusCode < 200 || statusCode > 399;
  }

  private void createHttpClientIfNecessary(boolean noTlsVerify) {
    if (httpClient != null) {
      return;
    }

    HttpClientBuilder builder = HttpClientBuilder.create();
    if (noTlsVerify) {
      try {
        SSLContextBuilder sslBuilder = new SSLContextBuilder()
          .loadTrustMaterial(null, new TrustSelfSignedStrategy());
        SSLConnectionSocketFactory connectionFactory =
          new SSLConnectionSocketFactory(sslBuilder.build(), new NoopHostnameVerifier());
        builder.setSSLSocketFactory(connectionFactory);
      } catch (NoSuchAlgorithmException | KeyStoreException | KeyManagementException e) {
        logger.warn("Could not set noTlsVerify to true, verification will remain", e);
      }
    }

    this.httpClient = builder.build();
  }
}
