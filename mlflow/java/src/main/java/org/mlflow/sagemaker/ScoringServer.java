package org.mlflow.sagemaker;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.mlflow.mleap.MLeapLoader;
import org.mlflow.models.Model;
import spark.Request;
import spark.Response;
import spark.Service;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.servlet.ServletContextHandler;

/** A RESTful webserver for {@link Predictor Predictors} that runs on the local host */
public class ScoringServer {
  public static final String RESPONSE_KEY_ERROR_MESSAGE = "Error";

  public static final int HTTP_RESPONSE_CODE_SERVER_ERROR = 500;
  public static final int HTTP_RESPONSE_CODE_SUCCESS = 200;
  public static final int DEFAULT_PORT = 5001;

  private enum RequestContentType {
    Csv("text/csv"),
    Json("application/json"),
    Invalid("invalid");

    private final String value;
    private static final Map<String, RequestContentType> BY_VALUE_MAP = new HashMap<>();

    static {
      for (RequestContentType inputType : RequestContentType.values()) {
        BY_VALUE_MAP.put(inputType.value, inputType);
      }
    }

    RequestContentType(String value) {
      this.value = value;
    }

    public static RequestContentType fromValue(String value) {
      if (BY_VALUE_MAP.containsKey(value)) {
        return BY_VALUE_MAP.get(value);
      } else {
        return RequestContentType.Invalid;
      }
    }
  }

  private final Server server;

  /**
   * Constructs a {@link ScoringServer} to serve the specified {@link Predictor} on the local host
   * at the default port: {@link ScoringServer#DEFAULT_PORT}
   */
  public ScoringServer(Predictor predictor) {
    this(predictor, DEFAULT_PORT);
  }

  /**
   * Constructs a {@link ScoringServer} to serve the specified {@link Predictor} on the local host
   * at the specified port
   */
  public ScoringServer(Predictor predictor, int portNumber) {
    this.server = new Server(portNumber);
    ServletContextHandler rootContextHandler = new ServletContextHandler(null, "/");
    rootContextHandler.addServlet(new ServletHolder(new ScoringServer.PingServlet()), "/ping");
    rootContextHandler.addServlet(
        new ServletHolder(new ScoringServer.InvocationsServlet(predictor)), "/invocations");
    this.server.setHandler(rootContextHandler);
  }

  /**
   * Loads the MLFlow model at the specified path as a {@link Predictor} and serves it on the local
   * host at the default port: {@link ScoringServer#DEFAULT_PORT}
   *
   * @param modelPath The path to the MLFlow model to serve
   */
  public ScoringServer(String modelPath) throws PredictorLoadingException {
    this(modelPath, DEFAULT_PORT);
  }

  /**
   * Loads the MLFlow model at the specified path as a {@link Predictor} and serves it on the local
   * host at the specified port
   *
   * @param modelPath The path to the MLFlow model to serve
   */
  public ScoringServer(String modelPath, int portNumber) throws PredictorLoadingException {
    this(loadPredictorFromPath(modelPath), portNumber);
  }

  private static Predictor loadPredictorFromPath(String modelPath)
      throws PredictorLoadingException {
    Model config = null;
    try {
      config = Model.fromRootPath(modelPath);
    } catch (IOException e) {
      throw new PredictorLoadingException(
          "Failed to load the configuration for the MLFlow model at the specified path.");
    }
    return (new MLeapLoader()).load(config);
  }

  /**
   * Starts the scoring server on the local host
   */
  public void start() throws Exception {
    this.server.start();
  }

  /**
   * Stops the scoring server
   */
  public void stop() throws Exception {
    this.server.stop();
  }

  /** @return `true` if the server is active (running), `false` otherwise */
  public boolean isActive() {
    return this.server.isStarted();
  }

  static class PingServlet extends HttpServlet {
    @Override
    public void doGet(HttpServletRequest request, HttpServletResponse response) {
      response.setStatus(HttpServletResponse.SC_OK);
    }
  }

  static class InvocationsServlet extends HttpServlet {
    private final Predictor predictor;

    InvocationsServlet(Predictor predictor) {
      this.predictor = predictor;
    }

    @Override
    public void doPost(HttpServletRequest request, HttpServletResponse response)
        throws IOException {
      RequestContentType contentType =
          RequestContentType.fromValue(request.getHeader("Content-type"));
      String requestBody =
          request.getReader().lines().collect(Collectors.joining(System.lineSeparator()));

      String responseContent = null;
      try {
        responseContent = evaluateRequest(requestBody, contentType);
      } catch (PredictorEvaluationException e) {
        response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        responseContent = getErrorResponseJson(e.getMessage());
      } catch (Exception e) {
        e.printStackTrace();
        response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        responseContent =
            getErrorResponseJson("An unknown error occurred while evaluating the model!");
      } finally {
        response.getWriter().print(responseContent);
        response.getWriter().close();
      }
    }

    private String evaluateRequest(String requestContent, RequestContentType contentType)
        throws PredictorEvaluationException {
      switch (contentType) {
        case Json: {
          DataFrame parsedInput = null;
          try {
            parsedInput = DataFrame.fromJson(requestContent);
          } catch (UnsupportedOperationException e) {
            throw new PredictorEvaluationException(
                "This model does not yet support evaluating JSON inputs.");
          }
          DataFrame result = predictor.predict(parsedInput);
          return result.toJson();
        }
        case Csv: {
          DataFrame parsedInput = null;
          try {
            parsedInput = DataFrame.fromCsv(requestContent);
          } catch (UnsupportedOperationException e) {
            throw new PredictorEvaluationException(
                "This model does not yet support evaluating CSV inputs.");
          }
          DataFrame result = predictor.predict(parsedInput);
          return result.toCsv();
        }
        default:
          throw new PredictorEvaluationException(
              "Invocations content must be of content type `application/json` or `text/csv`");
      }
    }

    private String getErrorResponseJson(String errorMessage) {
      String response =
          String.format("{ \"%s\" : \"%s\" }", RESPONSE_KEY_ERROR_MESSAGE, errorMessage);
      return response;
    }
  }

  /**
   * Entrypoint for locally serving MLFlow models with the MLeap flavor using the {@link
   * ScoringServer}
   *
   * <p>This entrypoint expects the following arguments: 1. The path to the MLFlow model to serve.
   * This model must have the MLeap flavor. 2. (Optional) the number of the port on which to serve
   * the MLFlow model.
   */
  public static void main(String[] args) throws IOException, PredictorLoadingException {
    String modelPath = args[0];
    Optional<Integer> portNum = Optional.empty();
    if (args.length > 1) {
      portNum = Optional.of(Integer.parseInt(args[2]));
    }
    ScoringServer server = new ScoringServer(modelPath, portNum.orElse(DEFAULT_PORT));
    try {
      server.start();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
