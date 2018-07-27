package com.databricks.mlflow.mleap;

import com.databricks.mlflow.Flavor;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.fasterxml.jackson.annotation.JsonProperty;

public class MLeapConfig implements Flavor {
    public static final String FLAVOR_NAME = "mleap";

    @JsonProperty("mleap_version") private String mleapVersion;
    @JsonProperty("model_data") private String modelDataPath;

    @Override
    public String getName() {
        return FLAVOR_NAME;
    }

    public String getMleapVersion() {
        return mleapVersion;
    }

    public String getModelDataPath() {
        return modelDataPath;
    }
}
