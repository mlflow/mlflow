package com.databricks.mlflow.models;

public class PredictorEvaluationException extends Exception {
    private final String message;

    public PredictorEvaluationException(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }
}
