package org.mlflow.sagemaker;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.servlet.http.HttpServletResponse;
import org.apache.commons.io.IOUtils;
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
  private class TestPredictor implements Predictor {
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

    @Override
    public PredictorDataWrapper predict(PredictorDataWrapper input)
        throws PredictorEvaluationException {
      if (succeed) {
        String responseText = this.responseContent.orElse("{ \"Text\" : \"Succeed!\" }");
        return new PredictorDataWrapper(responseText, PredictorDataWrapper.ContentType.Json);
      } else {
        throw new PredictorEvaluationException("Failure!");
      }
    }
  }

  private static final HttpClient httpClient = HttpClientBuilder.create().build();

  private static String getHttpResponseBody(HttpResponse response) throws IOException {
    return IOUtils.toString(response.getEntity().getContent(), StandardCharsets.UTF_8);
  }

  @Test
  public void testScoringServerWithValidPredictorRespondsToPingsCorrectly() throws IOException {
    TestPredictor validPredictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(validPredictor);
    server.start();

    String requestUrl = String.format("http://localhost:%d/ping", server.getPort().get());
    HttpGet getRequest = new HttpGet(requestUrl);
    HttpResponse response = httpClient.execute(getRequest);
    Assert.assertEquals(HttpServletResponse.SC_OK, response.getStatusLine().getStatusCode());
    server.stop();
  }

  @Test
  public void testScoringServerWithValidPredictorRespondsToVersionCorrectly() throws IOException {
    TestPredictor validPredictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(validPredictor);
    server.start();

    String requestUrl = String.format("http://localhost:%d/version", server.getPort().get());
    HttpGet getRequest = new HttpGet(requestUrl);
    HttpResponse response = httpClient.execute(getRequest);
    Assert.assertEquals(HttpServletResponse.SC_OK, response.getStatusLine().getStatusCode());
    String responseBody = getHttpResponseBody(response);
    Assert.assertEquals("2.11.1-SNAPSHOT", responseBody);
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
        HttpServletResponse.SC_BAD_REQUEST, response.getStatusLine().getStatusCode());

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
      Assert.assertEquals(HttpServletResponse.SC_OK, response.getStatusLine().getStatusCode());
    }

    for (ScoringServer server : servers) {
      server.stop();
    }
  }

  @Test
  public void testScoringServerWithValidPredictorRespondsToInvocationWithPredictorOutputContent()
      throws IOException {
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
    Assert.assertEquals(HttpServletResponse.SC_OK, response.getStatusLine().getStatusCode());
    String responseBody = getHttpResponseBody(response);
    Map<String, String> responseDict = SerializationUtils.fromJson(responseBody, Map.class);
    Assert.assertEquals(predictorDict, responseDict);

    server.stop();
  }

  @Test
  public void testScoringServerRespondsWithInternalServerErrorCodeWhenPredictorThrowsException()
      throws IOException {
    TestPredictor predictor = new TestPredictor(false);

    ScoringServer server = new ScoringServer(predictor);
    server.start();

    String requestUrl = String.format("http://localhost:%d/invocations", server.getPort().get());

    HttpPost postRequestJson = new HttpPost(requestUrl);
    postRequestJson.addHeader("Content-type", "application/json");
    HttpEntity entityJson = new StringEntity("body");
    postRequestJson.setEntity(entityJson);

    HttpResponse responseJson = httpClient.execute(postRequestJson);
    Assert.assertEquals(
        HttpServletResponse.SC_INTERNAL_SERVER_ERROR, responseJson.getStatusLine().getStatusCode());

    HttpPost postRequestCsv = new HttpPost(requestUrl);
    postRequestCsv.addHeader("Content-type", "text/csv");
    HttpEntity entityCsv = new StringEntity("body");
    postRequestCsv.setEntity(entityCsv);

    HttpResponse responseCsv = httpClient.execute(postRequestCsv);
    Assert.assertEquals(
        HttpServletResponse.SC_INTERNAL_SERVER_ERROR, responseCsv.getStatusLine().getStatusCode());

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
      Assert.assertEquals(HttpServletResponse.SC_OK, response1.getStatusLine().getStatusCode());

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
    Assert.assertEquals(true, portNumber.get() > 0);
    server.stop();
  }

  @Test
  public void testScoringServerIsActiveReturnsTrueWhenServerIsRunningElseFalse() {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    Assert.assertEquals(false, server.isActive());
    server.start();
    Assert.assertEquals(true, server.isActive());
    server.stop();
    Assert.assertEquals(false, server.isActive());
  }

  @Test
  public void testGetPortReturnsEmptyForInactiveServer() {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    Optional<Integer> portNumber = server.getPort();
    Assert.assertEquals(false, portNumber.isPresent());
  }

  @Test
  public void testGetPortReturnsPresentOptionalForActiveServer() {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    server.start();
    Assert.assertEquals(true, server.isActive());
    Optional<Integer> portNumber = server.getPort();
    Assert.assertEquals(true, portNumber.isPresent());
    server.stop();
  }

  @Test
  public void testServerStartsOnSpecifiedPortOrThrowsStateChangeException() throws IOException {
    int portNumber = 6783;
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server1 = new ScoringServer(predictor);
    server1.start(portNumber);
    Assert.assertEquals(true, server1.isActive());

    String requestUrl = String.format("http://localhost:%d/ping", server1.getPort().get());
    HttpGet getRequest = new HttpGet(requestUrl);
    HttpResponse response = httpClient.execute(getRequest);
    Assert.assertEquals(HttpServletResponse.SC_OK, response.getStatusLine().getStatusCode());

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
