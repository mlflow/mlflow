package com.databricks.mlflow.mleap;

import com.databricks.mlflow.Flavor;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.fasterxml.jackson.annotation.JsonProperty;

public class MLeapFlavor implements Flavor {
    public static final String FLAVOR_NAME = "mleap";

    @JsonProperty("mleap_version") private String mleapVersion;
    @JsonProperty("model_data") private String modelDataPath;
    @JsonProperty("input_schema") private String inputSchemaPath;

    @Override
    public String getName() {
        return FLAVOR_NAME;
    }

    @Override
    public String getModelDataPath() {
        return modelDataPath;
    }

    public String getMleapVersion() {
        return mleapVersion;
    }

    public String getInputSchemaPath() {
        return inputSchemaPath;
    }
}
