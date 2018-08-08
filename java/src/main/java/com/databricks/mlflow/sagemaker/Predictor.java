package com.databricks.mlflow.sagemaker;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public abstract class Predictor {
    protected abstract DataFrame predict(DataFrame input) throws PredictorEvaluationException;
}
