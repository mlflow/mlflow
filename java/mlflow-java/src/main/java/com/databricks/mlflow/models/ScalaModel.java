package com.databricks.mlflow.models;

public abstract class ScalaModel extends JavaModel {
    @Override public abstract String predict(String input);
}
