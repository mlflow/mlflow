package com.databricks.mlflow.javafunc;

import com.databricks.mlflow.Flavor;
import com.databricks.mlflow.TrackingUtils;
import com.databricks.mlflow.models.JavaModel;
import com.databricks.mlflow.models.ModelConfig;

import java.io.File;
import java.util.Optional;

public class JavaFunc implements Flavor {
    public static final String FLAVOR_NAME = "javafunc";

    @Override
    public String getName() {
        return FLAVOR_NAME;
    }

    public static JavaModel load(String path, Optional<String> runId) {
        if (runId.isPresent()) {
            // Get the run-relative model logging directory
            path = TrackingUtils.getModelLogDir(path, runId.get());
        }
        String configPath = path + File.separator + "MLmodel";
        ModelConfig config = ModelConfig.fromPath(configPath);
        Optional<JavaFunc> javaFuncFlavor = config.getFlavor(FLAVOR_NAME, JavaFunc.class);
        if (!javaFuncFlavor.isPresent()) {
            // throw new Exception();
        }

        return null;
    }
}
