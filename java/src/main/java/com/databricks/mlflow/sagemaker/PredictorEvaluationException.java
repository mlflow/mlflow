package com.databricks.mlflow.sagemaker;

public class PredictorEvaluationException extends Exception {
    private final String message;

    public PredictorEvaluationException(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }
}
