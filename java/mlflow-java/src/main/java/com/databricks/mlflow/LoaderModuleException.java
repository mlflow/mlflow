package com.databricks.mlflow;

public class LoaderModuleException extends Exception {
    private final String className;

    public LoaderModuleException(String className) {
        this.className = className;
    }

    public String getMessage() {
        return String.format(
            "An error occurred while invoking the loader module with class name %s",
            this.className);
    }
}
