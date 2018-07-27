package com.databricks.mlflow.models;

import com.databricks.mlflow.models.UnsupportedInputTypeException;

class TestModel extends JavaModel {
    @Override
    public String predict(String input) {
        System.out.println(input);
        return "sample_response";
    }

    public static void main(String[] args) {
        TestModel testModel = new TestModel();
        testModel.serve();
    }
}
