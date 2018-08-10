package com.databricks.mlflow.utils;

public class TrackingUtils {
    public static String getModelLogDir(String path, String runId)
        throws UnsupportedOperationException {
        throw new UnsupportedOperationException(
            "Loading models based on run ids is not yet supported!");
    }
}
