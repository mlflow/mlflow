package org.mlflow.sagemaker;

import org.junit.Assert;
import org.junit.Test;
import org.junit.After;

import org.mlflow.utils.SerializationUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.exceptions.UnirestException;

import com.fasterxml.jackson.core.JsonProcessingException;

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

  @After
  public void awaitServerShutdown() throws InterruptedException {
    Thread.sleep(5000);
  }

  @Test
  public void testScoringServerWithValidPredictorRespondsToPingsCorrectly()
      throws InterruptedException {
    TestPredictor validPredictor = new TestPredictor(true);
    int portNumber = 5001;
    ScoringServer server = new ScoringServer(validPredictor, portNumber);
    server.start();

    Thread.sleep(5000);

    String requestUrl = String.format("http://localhost:%d/ping", portNumber);
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
  public void testScoringServerWithMissingPredictorRespondsToPingsWithServerErrorCode()
      throws InterruptedException {
    String badModelPath = "/not/a/valid/path";
    int portNumber = 5001;
    ScoringServer server = new ScoringServer(badModelPath, portNumber, false);
    server.start();

    Thread.sleep(5000);

    String requestUrl = String.format("http://localhost:%d/ping", portNumber);
    try {
      HttpResponse response = Unirest.get(requestUrl).asJson();
      Assert.assertEquals(response.getStatus(), ScoringServer.HTTP_RESPONSE_CODE_SERVER_ERROR);
    } catch (UnirestException e) {
      e.printStackTrace();
      Assert.fail("Encountered an exception while attempting to ping the server!");
    }
    server.stop();
  }

  @Test
  public void testScoringServerWithMissingPredictorRespondsToInvocationsWithServerErrorCode()
      throws InterruptedException {
    String badModelPath = "/not/a/valid/path";
    int portNumber = 5001;
    ScoringServer server = new ScoringServer(badModelPath, portNumber, false);
    server.start();

    Thread.sleep(5000);

    String requestUrl = String.format("http://localhost:%d/invocations", portNumber);
    try {
      HttpResponse<JsonNode> response =
          Unirest.post(requestUrl).header("Content-type", "application/json").body("body").asJson();
      Assert.assertEquals(response.getStatus(), ScoringServer.HTTP_RESPONSE_CODE_SERVER_ERROR);
    } catch (UnirestException e) {
      e.printStackTrace();
      Assert.fail("Encountered an exception while attempting to ping the server!");
    }
    server.stop();
  }

  @Test
  public void testMultipleServersRunOnDifferentPortsSuccessfully() throws InterruptedException {
    List<Integer> portNumbers = Arrays.asList(5001, 5002, 5003);
    TestPredictor predictor = new TestPredictor(true);

    List<ScoringServer> servers = new ArrayList<>();
    for (int portNumber : portNumbers) {
      ScoringServer newServer = new ScoringServer(predictor, portNumber);
      newServer.start();
      servers.add(newServer);
    }

    Thread.sleep(5000);

    for (int portNumber : portNumbers) {
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
      throws InterruptedException, UnirestException, IOException, JsonProcessingException {
    Map<String, String> predictorDict = new HashMap<>();
    predictorDict.put("Text", "Response");
    String predictorJson = SerializationUtils.toJson(predictorDict);
    TestPredictor predictor = new TestPredictor(predictorJson);
    int portNumber = 5001;

    ScoringServer server = new ScoringServer(predictor, portNumber);
    server.start();

    Thread.sleep(5000);

    String requestUrl = String.format("http://localhost:%d/invocations", portNumber);
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
      throws InterruptedException, UnirestException, IOException {
    TestPredictor predictor = new TestPredictor(false);
    int portNumber = 5001;

    ScoringServer server = new ScoringServer(predictor, portNumber);
    server.start();

    Thread.sleep(5000);

    String requestUrl = String.format("http://localhost:%d/invocations", portNumber);
    HttpResponse<JsonNode> response =
        Unirest.post(requestUrl).header("Content-type", "application/json").body("body").asJson();
    Assert.assertEquals(response.getStatus(), ScoringServer.HTTP_RESPONSE_CODE_SERVER_ERROR);

    server.stop();
  }

  @Test
  public void testInactiveScoringServerThrowsIllegalStateExceptionWhenStopped() {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor, 5001);
    try {
      server.stop();
      Assert.fail(
          "Expected the server stop operation to throw an IllegalStateException, but none was thrown.");
    } catch (IllegalStateException e) {
      // Succeed
    }
  }

  @Test
  public void testActiveScoringServerThrowsIllegalStateExceptionWhenStarted()
      throws InterruptedException {
    TestPredictor predictor = new TestPredictor(true);
    ScoringServer server = new ScoringServer(predictor, 5001);
    server.start();

    Thread.sleep(5000);

    try {
      server.start();
      Assert.fail(
          "Expected the server stop operation to throw an IllegalStateException, but none was thrown.");
    } catch (IllegalStateException e) {
      // Succeed
    }
    server.stop();
  }

  @Test
  public void testScoringServerStartsAndStopsSuccessfully() throws InterruptedException {
    TestPredictor predictor = new TestPredictor(true);
    int portNumber = 5001;
    ScoringServer server = new ScoringServer(predictor, portNumber);
    String requestUrl = String.format("http://localhost:%d/ping", portNumber);

    for (int i = 0; i < 3; ++i) {
      server.start();
      Thread.sleep(5000);
      try {
        HttpResponse response = Unirest.get(requestUrl).asJson();
      } catch (UnirestException e) {
        Assert.fail(
            "Encountered an unexpected exception while attempting to ping the active scoring server.");
      }
      server.stop();
      Thread.sleep(5000);
      try {
        HttpResponse response = Unirest.get(requestUrl).asJson();
        Assert.fail("Expected the attempt to query an inactive server to throw an exception.");
      } catch (UnirestException e) {
        // Succeed
      }
    }
  }
}
