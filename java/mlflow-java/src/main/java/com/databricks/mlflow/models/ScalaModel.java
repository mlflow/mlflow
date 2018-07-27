package com.databricks.mlflow;

public abstract class ScalaModel extends JavaModel {
    @Override public abstract String predict(String input);
}
