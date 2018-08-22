package org.mlflow.sagemaker;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.Assert;
import org.junit.After;
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
        String responseText = this.responseContent.orElse("{ \"Text\" : \"Success!\" }");
        return DataFrame.fromJson(responseText);
      } else {
        throw new PredictorEvaluationException("Failure!");
      }
    }
  }

  @Test
  public void testScoringServerWithValidPredictorRespondsToPingsCorrectly() throws Exception {
    TestPredictor validPredictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(validPredictor);
    server.start();

    String requestUrl = String.format("http://localhost:%d/ping", server.getPort().get());
    try {
      HttpResponse response = Unirest.get(requestUrl).asJson();
      Assert.assertEquals(response.getStatus(), ScoringServer.HTTP_RESPONSE_CODE_SUCCESS);
    } catch (UnirestException e) {
      e.printStackTrace();
      Assert.fail("Encountered an exception while attempting to ping the server!");
    }
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
      // Success
    }

    try {
      ScoringServer server = new ScoringServer(badModelPath);
      Assert.fail("Expected constructing a model server with an invalid model path"
          + " to throw an exception, but none was thrown.");
    } catch (PredictorLoadingException e) {
      // Success
    }
  }

  @Test
  public void testScoringServerRepondsToInvocationOfBadContentTypeWithServerErrorCode()
      throws Exception {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    server.start();

    String requestUrl = String.format("http://localhost:%d/invocations", server.getPort().get());
    try {
      String badContentType = "not-a-content-type";
      HttpResponse<JsonNode> response =
          Unirest.post(requestUrl).header("Content-type", badContentType).body("body").asJson();
      Assert.assertEquals(response.getStatus(), ScoringServer.HTTP_RESPONSE_CODE_SERVER_ERROR);
    } catch (UnirestException e) {
      e.printStackTrace();
      Assert.fail("Encountered an exception while attempting to invoke the server!");
    } finally {
      server.stop();
    }
  }

  @Test
  public void testMultipleServersRunOnDifferentPortsSuccessfully() throws Exception {
    TestPredictor predictor = new TestPredictor(true);

    List<ScoringServer> servers = new ArrayList<>();
    for (int i = 0; i < 3; ++i) {
      ScoringServer newServer = new ScoringServer(predictor);
      newServer.start();
      servers.add(newServer);
    }

    for (ScoringServer server : servers) {
      int portNumber = server.getPort().get();
      try {
        String requestUrl = String.format("http://localhost:%d/ping", portNumber);
        HttpResponse response = Unirest.get(requestUrl).asJson();
        Assert.assertEquals(response.getStatus(), ScoringServer.HTTP_RESPONSE_CODE_SUCCESS);
      } catch (UnirestException e) {
        e.printStackTrace();
        Assert.fail(String.format(
            "Encountered an exception while attempting to ping the server on port %d!",
            portNumber));
      }
    }

    for (ScoringServer server : servers) {
      server.stop();
    }
  }

  @Test
  public void testScoringServerWithValidPredictorRespondsToInvocationWithPredictorOutputContent()
      throws Exception, UnirestException, IOException, JsonProcessingException {
    Map<String, String> predictorDict = new HashMap<>();
    predictorDict.put("Text", "Response");
    String predictorJson = SerializationUtils.toJson(predictorDict);
    TestPredictor predictor = new TestPredictor(predictorJson);

    ScoringServer server = new ScoringServer(predictor);
    server.start();

    String requestUrl = String.format("http://localhost:%d/invocations", server.getPort().get());
    HttpResponse<JsonNode> response =
        Unirest.post(requestUrl).header("Content-type", "application/json").body("body").asJson();
    Assert.assertEquals(response.getStatus(), ScoringServer.HTTP_RESPONSE_CODE_SUCCESS);
    String responseJson = response.getBody().toString();
    Map<String, String> responseDict = SerializationUtils.fromJson(responseJson, Map.class);
    Assert.assertEquals(responseDict, predictorDict);

    server.stop();
  }

  @Test
  public void testScoringServerRespondsWithServerErrorCodeWhenPredictorThrowsException()
      throws Exception, UnirestException, IOException {
    TestPredictor predictor = new TestPredictor(false);

    ScoringServer server = new ScoringServer(predictor);
    server.start();

    String requestUrl = String.format("http://localhost:%d/invocations", server.getPort().get());
    HttpResponse<JsonNode> response =
        Unirest.post(requestUrl).header("Content-type", "application/json").body("body").asJson();
    Assert.assertEquals(response.getStatus(), ScoringServer.HTTP_RESPONSE_CODE_SERVER_ERROR);

    server.stop();
  }

  @Test
  public void testScoringServerStartsAndStopsSuccessfully() throws Exception {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);

    for (int i = 0; i < 3; ++i) {
      server.start();
      String requestUrl = String.format("http://localhost:%d/ping", server.getPort().get());

      try {
        HttpResponse response = Unirest.get(requestUrl).asJson();
      } catch (UnirestException e) {
        Assert.fail("Encountered an unexpected exception while attempting to ping"
            + "the active scoring server.");
      } finally {
        server.stop();
      }

      try {
        HttpResponse response = Unirest.get(requestUrl).asJson();
        Assert.fail("Expected the attempt to query an inactive server to throw an exception.");
      } catch (UnirestException e) {
        // Succeed
      }
    }
  }

  @Test
  public void testScoringServerIsActiveReturnsTrueWhenServerIsRunningElseFalse() throws Exception {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    Assert.assertEquals(server.isActive(), false);
    server.start();

    Assert.assertEquals(server.isActive(), true);
    server.stop();
    Assert.assertEquals(server.isActive(), false);
  }

  @Test
  public void testGetPortReturnsEmptyForInactiveServer() throws Exception {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    Optional<Integer> portNumber = server.getPort();
    Assert.assertEquals(portNumber.isPresent(), false);
  }

  @Test
  public void testGetPortReturnsPresentOptionalForActiveServer() throws Exception {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor);
    server.start();
    Assert.assertEquals(server.isActive(), true);
    Optional<Integer> portNumber = server.getPort();
    Assert.assertEquals(portNumber.isPresent(), true);
    server.stop();
  }

  @Test
  public void testServerStartsOnSpecifiedPortOrThrows() throws Exception {
    int portNumber = 6783;
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server1 = new ScoringServer(predictor);
    server1.start(portNumber);
    Assert.assertEquals(server1.isActive(), true);

    ScoringServer server2 = new ScoringServer(predictor);
    try {
      server2.start(portNumber);
      Assert.fail(
          "Expected starting a new server on a port that is already bound to throw an exception.");
    } catch (Exception e) {
      // Success
    }
    server1.stop();
    server2.stop();
  }
}
