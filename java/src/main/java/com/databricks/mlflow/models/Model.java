package com.databricks.mlflow.models;

import com.databricks.mlflow.Flavor;
import com.databricks.mlflow.utils.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Optional;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

public class Model {
    @JsonProperty("artifact_path") private String artifactPath;
    @JsonProperty("run_id") private String runId;
    @JsonProperty("utc_time_created") private String utcTimeCreated;
    @JsonProperty("flavors") private Map<String, Object> flavors;

    private String rootPath;

    public static Model fromRootPath(String modelRootPath) throws IOException {
        String configPath = FileUtils.join(modelRootPath, "MLmodel");
        return fromConfigPath(configPath);
    }

    public static Model fromConfigPath(String configPath) throws IOException {
        File configFile = new File(configPath);
        Model model = FileUtils.parseYamlFromFile(configFile, Model.class);
        // Set the base path to the directory containing the configuration file.
        // This will be used to create an absolute path to the serialized model
        model.setRootPath(configFile.getParentFile().getAbsolutePath());
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

    public Optional<String> getRootPath() {
        return convertFieldToOptional(this.rootPath);
    }

    private Optional<String> convertFieldToOptional(String field) {
        if (field != null) {
            return Optional.of(field);
        } else {
            return Optional.<String>empty();
        }
    }

    private void setRootPath(String rootPath) {
        this.rootPath = rootPath;
    }
}
