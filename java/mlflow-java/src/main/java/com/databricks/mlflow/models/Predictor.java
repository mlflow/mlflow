package com.databricks.mlflow.models;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public abstract class Predictor {
    public abstract String predict(String input) throws PredictorEvaluationException;
}
