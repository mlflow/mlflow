package com.databricks.mlflow.models;

class UnsupportedInputTypeException extends PredictorEvaluationException {
    protected UnsupportedInputTypeException() {
        super("Unsupported request input type!");
    }
}
