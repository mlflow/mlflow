package com.databricks.mlflow.models;

public class ModelEvaluationException extends Exception {
    private final String message;

    public ModelEvaluationException(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }
}
