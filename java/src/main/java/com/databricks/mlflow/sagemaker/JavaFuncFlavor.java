package com.databricks.mlflow.sagemaker;

import com.databricks.mlflow.Flavor;

import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * NOTE: This module is experimental and subject to change, pending investigation of the
 * appropriate input format for java function objects. It provides functionality that parses
 * a reasonable configuration format for serialized Java models. This configuration has not
 * yet been solidified and is also subject to change.
 */
public class JavaFuncFlavor implements Flavor {
    public static final String FLAVOR_NAME = "java_function";

    @JsonProperty("data") private String modelDataPath;
    @JsonProperty("code") private String codePath;
    @JsonProperty("packages") private List<String> packageDependencies;
    @JsonProperty("loader_module") private String loaderClassName;

    @Override
    public String getName() {
        return FLAVOR_NAME;
    }

    @Override
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
