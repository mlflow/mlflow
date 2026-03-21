package org.mlflow.tracking;

import com.google.common.annotations.VisibleForTesting;

import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;

import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpEntityEnclosingRequestBase;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPatch;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpRequestBase;
import org.apache.http.conn.ssl.NoopHostnameVerifier;
import org.apache.http.conn.ssl.SSLConnectionSocketFactory;
import org.apache.http.conn.ssl.TrustSelfSignedStrategy;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.util.EntityUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.mlflow.tracking.creds.MlflowHostCreds;
import org.mlflow.tracking.creds.MlflowHostCredsProvider;


class MlflowHttpCaller {
  private static final Logger logger = LoggerFactory.getLogger(MlflowHttpCaller.class);
  private static final String BASE_API_PATH = "api/2.0/mlflow";
  protected CloseableHttpClient httpClient;
  private final MlflowHostCredsProvider hostCredsProvider;
  private final int maxRateLimitIntervalMillis;
  private final int rateLimitRetrySleepInitMillis;
  private final int maxRetryAttempts;

  /**
   * Construct a new MlflowHttpCaller with a default configuration for request retries.
   */
  MlflowHttpCaller(MlflowHostCredsProvider hostCredsProvider) {
    this(hostCredsProvider, 60000, 1000, 3);
  }

  /**
   * Construct a new MlflowHttpCaller.
   *
   * @param maxRateLimitIntervalMs The maximum amount of time, in milliseconds, to spend retrying a
   *                               single request in response to rate limiting (error code 429).
   * @param rateLimitRetrySleepInitMs The initial backoff delay, in milliseconds, when retrying a
   *                                  request in response to rate limiting (error code 429). The
   *                                  delay is increased exponentially after each rate limiting
   *                                  response until the total delay incurred across all retries for
   *                                  the request exceeds the specified maxRateLimitIntervalSeconds.
   * @param maxRetryAttempts The maximum number of times to retry a request, excluding rate limit
   *                         retries.
   */
  MlflowHttpCaller(MlflowHostCredsProvider hostCredsProvider,
                   int maxRateLimitIntervalMs,
                   int rateLimitRetrySleepInitMs,
                   int maxRetryAttempts) {
    this.hostCredsProvider = hostCredsProvider;
    this.maxRateLimitIntervalMillis = maxRateLimitIntervalMs;
    this.rateLimitRetrySleepInitMillis = rateLimitRetrySleepInitMs;
    this.maxRetryAttempts = maxRetryAttempts;
  }

  @VisibleForTesting
  MlflowHttpCaller(MlflowHostCredsProvider hostCredsProvider,
                   int maxRateLimitIntervalMs,
                   int rateLimitRetrySleepInitMs,
                   int maxRetryAttempts,
                   CloseableHttpClient client) {
    this(
      hostCredsProvider, maxRateLimitIntervalMs, rateLimitRetrySleepInitMs, maxRetryAttempts);
    this.httpClient = client;
  }

  private HttpResponse executeRequestWithRateLimitRetries(HttpRequestBase request)
      throws IOException {
    int timeLeft = maxRateLimitIntervalMillis;
    int sleepFor = rateLimitRetrySleepInitMillis;
    HttpResponse response = httpClient.execute(request);
    while (response.getStatusLine().getStatusCode() == 429 && timeLeft > 0) {
      logger.warn("Request returned with status code 429 (Rate limit exceeded). Retrying after "
                  + sleepFor
                  + " milliseconds. Will continue to retry 429s for up to "
                  + timeLeft
                  + " milliseconds.");
      try {
        Thread.sleep(sleepFor);
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
      timeLeft -= sleepFor;
      sleepFor = Math.min(timeLeft, 2 * sleepFor);
      response = httpClient.execute(request);
    }
    checkError(response);
    return response;
  }

  private HttpResponse executeRequest(HttpRequestBase request) throws IOException {
    HttpResponse response = null;
    int attemptsRemaining = this.maxRetryAttempts;
    while (attemptsRemaining > 0) {
      attemptsRemaining -= 1;
      try {
        response = executeRequestWithRateLimitRetries(request);
        break;
      } catch (MlflowHttpException e) {
        if (attemptsRemaining > 0 && e.getStatusCode() != 429) {
          logger.warn("Request returned with status code {} (Rate limit exceeded)."
                      + " Retrying up to {} more times. Response body: {}",
                      e.getStatusCode(),
                      attemptsRemaining,
                      e.getBodyMessage());
          continue;
        } else {
          throw e;
        }
      }
    }
    return response;
  }

  String get(String path) {
    logger.debug("Sending GET " + path);
    HttpGet request = new HttpGet();
    fillRequestSettings(request, path);
    try {
      HttpResponse response = executeRequest(request);
      String responseJson = EntityUtils.toString(response.getEntity(), StandardCharsets.UTF_8);
      logger.debug("Response: " + responseJson);
      return responseJson;
    } catch (IOException e) {
      throw new MlflowClientException(e);
    }
  }

  // TODO(aaron) Convert to InputStream.
  byte[] getAsBytes(String path) {
    logger.debug("Sending GET " + path);
    HttpGet request = new HttpGet();
    fillRequestSettings(request, path);
    try {
      HttpResponse response = executeRequest(request);
      byte[] bytes = EntityUtils.toByteArray(response.getEntity());
      logger.debug("response: #bytes=" + bytes.length);
      return bytes;
    } catch (IOException e) {
      throw new MlflowClientException(e);
    }
  }

  String post(String path, String json) {
    logger.debug("Sending POST " + path + ": " + json);
    HttpPost request = new HttpPost();
    return send(request, path, json);
  }

  String patch(String path, String json) {
    logger.debug("Sending PATCH " + path + ": " + json);
    HttpPatch request = new HttpPatch();
    return send(request, path, json);
  }

  private String send(HttpEntityEnclosingRequestBase request, String path, String json) {
    fillRequestSettings(request, path);
    request.setEntity(new StringEntity(json, StandardCharsets.UTF_8));
    request.setHeader("Content-Type", "application/json");
    try {
      HttpResponse response = executeRequest(request);
      String responseJson = EntityUtils.toString(response.getEntity(), StandardCharsets.UTF_8);
      logger.debug("Response: " + responseJson);
      return responseJson;
    } catch (IOException e) {
      throw new MlflowClientException(e);
    }
  }

  private void checkError(HttpResponse response) throws MlflowClientException, IOException {
    int statusCode = response.getStatusLine().getStatusCode();
    String reasonPhrase = response.getStatusLine().getReasonPhrase();
    if (isError(statusCode)) {
      String bodyMessage = EntityUtils.toString(response.getEntity(), StandardCharsets.UTF_8);
      if (statusCode >= 400 && statusCode <= 499) {
        throw new MlflowHttpException(statusCode, reasonPhrase, bodyMessage);
      }
      if (statusCode >= 500 && statusCode <= 599) {
        throw new MlflowHttpException(statusCode, reasonPhrase, bodyMessage);
      }
      throw new MlflowHttpException(statusCode, reasonPhrase, bodyMessage);
    }
  }

  private void fillRequestSettings(HttpRequestBase request, String path) {
    MlflowHostCreds hostCreds = hostCredsProvider.getHostCreds();
    createHttpClientIfNecessary(hostCreds.shouldIgnoreTlsVerification());
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

    String userAgent = "mlflow-java-client";
    String clientVersion = MlflowClientVersion.getClientVersion();
    if (!clientVersion.isEmpty()) {
      userAgent += "/" + clientVersion;
    }
    request.addHeader("User-Agent", userAgent);
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

  void close() {
    if (httpClient != null) {
      try {
	  httpClient.close();
      } catch(IOException e){
	  logger.warn("Unable to close connection to mlflow backend", e);
      }
    }
  }
}
