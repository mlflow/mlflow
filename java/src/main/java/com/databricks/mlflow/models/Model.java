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

    private String basePath;

    public static Model fromRootPath(String modelRootPath) throws IOException {
        String configPath = modelRootPath + File.separator + "MLmodel";
        return fromConfigPath(configPath);
    }

    public static Model fromConfigPath(String configPath) throws IOException {
        File configFile = new File(configPath);
        final ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        Model model = mapper.readValue(configFile, Model.class);
        // Set the base path to the directory containing the configuration file.
        // This will be used to create an absolute path to the serialized model
        model.setBasePath(configFile.getParentFile().getAbsolutePath());
        return model;
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

    public Optional<String> getBasePath() {
        return convertFieldToOptional(this.basePath);
    }

    private Optional<String> convertFieldToOptional(String field) {
        if (field != null) {
            return Optional.of(field);
        } else {
            return Optional.<String>empty();
        }
    }

    private void setBasePath(String basePath) {
        this.basePath = basePath;
    }
}
