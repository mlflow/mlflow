package com.databricks.mlflow.models;

import com.databricks.mlflow.Flavor;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Optional;
import java.io.File;
import java.io.IOException;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import com.fasterxml.jackson.annotation.JsonProperty;

public class Model {
    @JsonProperty("artifact_path") private String artifactPath;
    @JsonProperty("run_id") private String runId;
    @JsonProperty("utc_time_created") private String utcTimeCreated;
    @JsonProperty("flavors") private Map<String, Object> flavors;

    public static Model fromPath(String configPath) throws IOException {
        final ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        return mapper.readValue(new File(configPath), Model.class);
    }

    public Optional<String> getArtifactPath() {
        return convertFieldToOptional(this.artifactPath);
    }

    public Optional<String> getUtcTimeCreated() {
        return convertFieldToOptional(this.utcTimeCreated);
    }

    public Optional<String> getRunId() {
        return convertFieldToOptional(this.runId);
    }

    public <T extends Flavor> Optional<T> getFlavor(String flavorName, Class<T> flavorClass) {
        if (this.flavors.containsKey(flavorName)) {
            final ObjectMapper mapper = new ObjectMapper();
            T flavor = mapper.convertValue(this.flavors.get(flavorName), flavorClass);
            return Optional.of(flavor);
        } else {
            return Optional.<T>empty();
        }
    }

    private Optional<String> convertFieldToOptional(String field) {
        if (field != null) {
            return Optional.of(field);
        } else {
            return Optional.<String>empty();
        }
    }
}
