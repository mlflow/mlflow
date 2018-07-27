package com.databricks.mlflow.javafunc;

import com.databricks.mlflow.Flavor;

import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.fasterxml.jackson.annotation.JsonProperty;

public class JavaFuncConfig implements Flavor {
    public static final String FLAVOR_NAME = "java_function";

    @JsonProperty("data") private String modelDataPath;
    @JsonProperty("code") private String codePath;
    @JsonProperty("packages") private List<String> packageDependencies;
    @JsonProperty("loader_module") private String loaderClassName;

    @Override
    public String getName() {
        return FLAVOR_NAME;
    }

    public String getModelDataPath() {
        return modelDataPath;
    }

    public String getCodePath() {
        return codePath;
    }

    public String getLoaderClassName() {
        return loaderClassName;
    }

    public List<String> getPackageDependencies() {
        return packageDependencies;
    }
}
