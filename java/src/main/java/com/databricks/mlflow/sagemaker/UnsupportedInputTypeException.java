package com.databricks.mlflow.sagemaker;

class UnsupportedInputTypeException extends PredictorEvaluationException {
    protected UnsupportedInputTypeException() {
        super("Unsupported request input type!");
    }
}
