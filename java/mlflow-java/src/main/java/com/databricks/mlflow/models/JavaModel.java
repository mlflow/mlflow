package com.databricks.mlflow.models;

import java.util.Map;
import java.util.HashMap;

import static spark.Spark.*;
import spark.Request;

public abstract class JavaModel {
    private enum RequestInputType {
        Csv("text/csv"),
        Json("application/json"),
        Invalid("invalid");

        private final String value;
        private static final Map<String, RequestInputType> BY_VALUE_MAP = new HashMap<>();
        static {
            for (RequestInputType inputType : RequestInputType.values()) {
                BY_VALUE_MAP.put(inputType.value, inputType);
            }
        }

        RequestInputType(String value) {
            this.value = value;
        }

        public static RequestInputType fromValue(String value) {
            if (BY_VALUE_MAP.containsKey(value)) {
                return BY_VALUE_MAP.get(value);
            } else {
                return RequestInputType.Invalid;
            }
        }
    }

    private static final int PORT_NUMBER = 8080;

    public abstract String predict(String input);

    public void serve() {
        port(PORT_NUMBER);
        get("/ping", (request, response) -> {
            response.status(200);
            return "";
        });

        post("/invocations", (request, response) -> {
            try {
                String result = evaluateRequest(request);
                response.status(200);
                return result;
            } catch (ModelEvaluationException e) {
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

    private String evaluateRequest(Request request) throws ModelEvaluationException {
        RequestInputType inputType = RequestInputType.fromValue(request.contentType());
        String parsedInput = null;
        switch (inputType) {
            case Json:
                // TODO: Do something case-specific
                parsedInput = request.body();
                break;
            case Csv:
                // TODO: Do something case-specific
                parsedInput = request.body();
                break;
            case Invalid:
            default:
                throw new UnsupportedInputTypeException();
        }
        return predict(parsedInput);
    }

    private String getErrorResponseJson(String errorMessage) {
        // TODO: Make this JSON-formatted
        return errorMessage;
    }
}
