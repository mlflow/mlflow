package com.databricks.mlflow.sagemaker;

import com.databricks.mlflow.LoaderModuleException;
import com.databricks.mlflow.javafunc.JavaFunc;
import com.databricks.mlflow.models.Predictor;
import com.databricks.mlflow.models.PredictorEvaluationException;

import java.util.Optional;
import java.util.Map;
import java.util.HashMap;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

import spark.Request;
import spark.Response;
import static spark.Spark.*;

public class SageMakerServer {
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

    private static final int DEFAULT_PORT = 8080;

    public static void serve(Predictor predictor, Optional<Integer> portNumber) {
        serve(Optional.of(predictor), portNumber);
    }

    public static void serve(
        String modelPath, Optional<String> runId, Optional<Integer> portNumber) {
        Optional<Predictor> predictor = Optional.empty();
        try {
            predictor = Optional.of(JavaFunc.load(modelPath, runId));
        } catch (InstantiationException | InvocationTargetException | LoaderModuleException
            | IOException e) {
            e.printStackTrace();
        }
        serve(predictor, portNumber);
    }

    private static void serve(Optional<Predictor> predictor, Optional<Integer> portNumber) {
        port(portNumber.orElse(DEFAULT_PORT));
        get("/ping", (request, response) -> {
            if (!predictor.isPresent()) {
                yieldMissingPredictorError(response);
            }
            response.status(200);
            return "";
        });

        post("/invocations", (request, response) -> {
            if (!predictor.isPresent()) {
                return yieldMissingPredictorError(response);
            }

            try {
                String result = evaluateRequest(predictor.get(), request);
                response.status(200);
                return result;
            } catch (PredictorEvaluationException e) {
                response.status(500);
                String errorMessage = e.getMessage();
                return getErrorResponseJson(errorMessage);
            } catch (Exception e) {
                e.printStackTrace();
                response.status(500);
                String errorMessage = "An unknown error occurred while evaluating the model!";
                return getErrorResponseJson(errorMessage);
            }
        });
    }

    private static String yieldMissingPredictorError(Response response) {
        response.status(500);
        return getErrorResponseJson("Error loading predictor! See container logs for details!");
    }

    private static String evaluateRequest(Predictor predictor, Request request)
        throws PredictorEvaluationException {
        RequestContentType inputType = RequestContentType.fromValue(request.contentType());
        Optional<String> parsedInput = Optional.<String>empty();
        switch (inputType) {
            case Json:
                // TODO: Do something case-specific
                parsedInput = Optional.of(request.body());
                break;
            case Csv:
                // TODO: Do something case-specific
                parsedInput = Optional.of(request.body());
                break;
            case Invalid:
            default:
                throw new UnsupportedContentTypeException(request.contentType());
        }
        return predictor.predict(parsedInput.get());
    }

    private static String getErrorResponseJson(String errorMessage) {
        // TODO: Make this JSON-formatted
        return errorMessage;
    }

    static class UnsupportedContentTypeException extends PredictorEvaluationException {
        protected UnsupportedContentTypeException(String contentType) {
            super(String.format("Unsupported request input type: %s", contentType));
        }
    }

    public static void main(String[] args) {
        String modelPath = args[0];
        Optional<String> runId = Optional.empty();
        Optional<Integer> portNum = Optional.empty();
        if (args.length > 1) {
            runId = Optional.of(args[1]);
        }
        if (args.length > 2) {
            portNum = Optional.of(Integer.parseInt(args[2]));
        }
        serve(modelPath, runId, portNum);
    }
}
