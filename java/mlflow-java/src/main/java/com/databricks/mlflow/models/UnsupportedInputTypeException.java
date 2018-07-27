package com.databricks.mlflow.models;

class UnsupportedInputTypeException extends ModelEvaluationException {
    protected UnsupportedInputTypeException() {
        super("Unsupported request input type!");
    }
}
