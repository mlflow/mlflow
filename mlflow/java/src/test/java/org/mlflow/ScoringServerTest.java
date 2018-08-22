package org.mlflow.sagemaker;

import com.fasterxml.jackson.core.JsonProcessingException;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import javax.servlet.http.HttpServletResponse;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.conn.HttpHostConnectException;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClientBuilder;
import org.junit.Assert;
import org.junit.Test;
import org.mlflow.utils.SerializationUtils;

public class ScoringServerTest {
  private class TestPredictor extends Predictor {
    private final boolean succeed;
    private final Optional<String> responseContent;

    private TestPredictor(boolean succeed) {
      this.succeed = succeed;
      this.responseContent = Optional.empty();
    }

    private TestPredictor(String responseContent) {
      this.responseContent = Optional.of(responseContent);
      this.succeed = true;
    }

    protected DataFrame predict(DataFrame input) throws PredictorEvaluationException {
      if (succeed) {
        String responseText = this.responseContent.orElse("{ \"Text\" : \"Succeed!\" }");
        return DataFrame.fromJson(responseText);
      } else {
        throw new PredictorEvaluationException("Failure!");
      }
    }
  }

  private static final HttpClient httpClient = HttpClientBuilder.create().build();

  private static String getHttpResponseBody(HttpResponse response) throws IOException {
    InputStream responseContentStream = response.getEntity().getContent();
    String body = new BufferedReader(new InputStreamReader(responseContentStream))
                      .lines()
                      .collect(Collectors.joining(System.lineSeparator()));
    return body;
  }

  @Test
  public void testScoringServerWithValidPredictorRespondsToPingsCorrectly() throws IOException {
    TestPredictor validPredictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(validPredictor);
    server.start();

    String requestUrl = String.format("http://localhost:%d/ping", server.getPort().get());
    HttpGet getRequest = new HttpGet(requestUrl);
    HttpResponse response = httpClient.execute(getRequest);
    Assert.assertEquals(response.getStatusLine().getStatusCode(), HttpServletResponse.SC_OK);
    server.stop();
  }

  @Test
  public void testConstructingScoringServerFromInvalidModelPathThrowsException() {
    String badModelPath = "/not/a/valid/path";
    try {
      ScoringServer server = new ScoringServer(badModelPath);
      Assert.fail("Expected constructing a model server with an invalid model path"
          + " to throw an exception, but none was thrown.");
    } catch (PredictorLoadingException e) {
      // Succeed
    }

    try {
      ScoringServer server = new ScoringServer(badModelPath);
      Assert.fail("Expected constructing a model server with an invalid model path"
          + " to throw an exception, but none was thrown.");
    } catch (PredictorLoadingException e) {
      // Succeed
    }
  }

  @Test
  public void testScoringServerRepondsToInvocationOfBadContentTypeWithBadRequestCode()
      throws IOException {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    server.start();

    String requestUrl = String.format("http://localhost:%d/invocations", server.getPort().get());
    String badContentType = "not-a-content-type";
    HttpPost postRequest = new HttpPost(requestUrl);
    postRequest.addHeader("Content-type", badContentType);
    HttpEntity entity = new StringEntity("body");
    postRequest.setEntity(entity);

    HttpResponse response = httpClient.execute(postRequest);
    Assert.assertEquals(
        response.getStatusLine().getStatusCode(), HttpServletResponse.SC_BAD_REQUEST);

    server.stop();
  }

  @Test
  public void testMultipleServersRunOnDifferentPortsSucceedfully() throws IOException {
    TestPredictor predictor = new TestPredictor(true);

    List<ScoringServer> servers = new ArrayList<>();
    for (int i = 0; i < 3; ++i) {
      ScoringServer newServer = new ScoringServer(predictor);
      newServer.start();
      servers.add(newServer);
    }

    for (ScoringServer server : servers) {
      int portNumber = server.getPort().get();
      String requestUrl = String.format("http://localhost:%d/ping", portNumber);
      HttpGet getRequest = new HttpGet(requestUrl);
      HttpResponse response = httpClient.execute(getRequest);
      Assert.assertEquals(response.getStatusLine().getStatusCode(), HttpServletResponse.SC_OK);
    }

    for (ScoringServer server : servers) {
      server.stop();
    }
  }

  @Test
  public void testScoringServerWithValidPredictorRespondsToInvocationWithPredictorOutputContent()
      throws IOException, JsonProcessingException {
    Map<String, String> predictorDict = new HashMap<>();
    predictorDict.put("Text", "Response");
    String predictorJson = SerializationUtils.toJson(predictorDict);
    TestPredictor predictor = new TestPredictor(predictorJson);

    ScoringServer server = new ScoringServer(predictor);
    server.start();

    String requestUrl = String.format("http://localhost:%d/invocations", server.getPort().get());
    HttpPost postRequest = new HttpPost(requestUrl);
    postRequest.addHeader("Content-type", "application/json");
    HttpEntity entity = new StringEntity("body");
    postRequest.setEntity(entity);

    HttpResponse response = httpClient.execute(postRequest);
    Assert.assertEquals(response.getStatusLine().getStatusCode(), HttpServletResponse.SC_OK);
    String responseBody = getHttpResponseBody(response);
    Map<String, String> responseDict = SerializationUtils.fromJson(responseBody, Map.class);
    Assert.assertEquals(responseDict, predictorDict);

    server.stop();
  }

  @Test
  public void testScoringServerRespondsWithInternalServerErrorCodeWhenPredictorThrowsException()
      throws IOException {
    TestPredictor predictor = new TestPredictor(false);

    ScoringServer server = new ScoringServer(predictor);
    server.start();

    String requestUrl = String.format("http://localhost:%d/invocations", server.getPort().get());
    HttpPost postRequest = new HttpPost(requestUrl);
    postRequest.addHeader("Content-type", "application/json");
    HttpEntity entity = new StringEntity("body");
    postRequest.setEntity(entity);

    HttpResponse response = httpClient.execute(postRequest);
    Assert.assertEquals(
        response.getStatusLine().getStatusCode(), HttpServletResponse.SC_INTERNAL_SERVER_ERROR);

    server.stop();
  }

  @Test
  public void testScoringServerStartsAndStopsSucceedfully() throws IOException {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);

    for (int i = 0; i < 3; ++i) {
      server.start();
      String requestUrl = String.format("http://localhost:%d/ping", server.getPort().get());
      HttpGet getRequest = new HttpGet(requestUrl);
      HttpResponse response1 = httpClient.execute(getRequest);
      Assert.assertEquals(response1.getStatusLine().getStatusCode(), HttpServletResponse.SC_OK);

      server.stop();

      try {
        HttpResponse response2 = httpClient.execute(getRequest);
        Assert.fail("Expected attempt to ping an inactive server to throw an exception.");
      } catch (HttpHostConnectException e) {
        // Succeed
      }
    }
  }

  @Test
  public void testStartingScoringServerOnRandomPortAssignsNonZeroPort() {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    server.start();
    Optional<Integer> portNumber = server.getPort();
    Assert.assertEquals(portNumber.get() > 0, true);
    server.stop();
  }

  @Test
  public void testScoringServerIsActiveReturnsTrueWhenServerIsRunningElseFalse() {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    Assert.assertEquals(server.isActive(), false);
    server.start();
    Assert.assertEquals(server.isActive(), true);
    server.stop();
    Assert.assertEquals(server.isActive(), false);
  }

  @Test
  public void testGetPortReturnsEmptyForInactiveServer() {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    Optional<Integer> portNumber = server.getPort();
    Assert.assertEquals(portNumber.isPresent(), false);
  }

  @Test
  public void testGetPortReturnsPresentOptionalForActiveServer() {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    server.start();
    Assert.assertEquals(server.isActive(), true);
    Optional<Integer> portNumber = server.getPort();
    Assert.assertEquals(portNumber.isPresent(), true);
    server.stop();
  }

  @Test
  public void testServerStartsOnSpecifiedPortOrThrowsStateChangeException() throws IOException {
    int portNumber = 6783;
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server1 = new ScoringServer(predictor);
    server1.start(portNumber);
    Assert.assertEquals(server1.isActive(), true);

    String requestUrl = String.format("http://localhost:%d/ping", server1.getPort().get());
    HttpGet getRequest = new HttpGet(requestUrl);
    HttpResponse response = httpClient.execute(getRequest);
    Assert.assertEquals(response.getStatusLine().getStatusCode(), HttpServletResponse.SC_OK);

    ScoringServer server2 = new ScoringServer(predictor);
    try {
      server2.start(portNumber);
      Assert.fail(
          "Expected starting a new server on a port that is already bound to throw an exception.");
    } catch (ScoringServer.ServerStateChangeException e) {
      // Succeed
    }
    server1.stop();
    server2.stop();
  }

  @Test
  public void testAttemptingToStartActiveServerThrowsIllegalStateException() {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    server.start();
    try {
      server.start();
      Assert.fail(
          "Expected attempt to start a server that is already active to throw an exception.");
    } catch (IllegalStateException e) {
      // Succeed
    } finally {
      server.stop();
    }
  }
}
