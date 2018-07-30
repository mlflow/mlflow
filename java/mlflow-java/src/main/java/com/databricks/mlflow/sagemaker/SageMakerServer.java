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

import static spark.Spark.*;
import spark.Request;

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

    public static void serve(Predictor predictor, int portNumber) {
        port(portNumber);
        get("/ping", (request, response) -> {
            response.status(200);
            return "";
        });

        post("/invocations", (request, response) -> {
            try {
                String result = evaluateRequest(predictor, request);
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

    public static void serve(String modelPath, Optional<String> runId, int port) {
        Optional<Predictor> predictor = Optional.empty();
        try {
            predictor = Optional.of(JavaFunc.load(modelPath, runId));
        } catch (InvocationTargetException | LoaderModuleException | IOException e) {
            e.printStackTrace();
        }
        serve(predictor, port);
    }

    private static void serve(Optional<Predictor> predictor, int port) {
        serve(predictor, port);
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
}
