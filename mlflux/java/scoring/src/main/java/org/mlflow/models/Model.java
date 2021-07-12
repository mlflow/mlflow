package org.mlflow.models;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import org.mlflow.Flavor;
import org.mlflow.utils.FileUtils;
import org.mlflow.utils.SerializationUtils;

/**
 * Represents an MLflow model. This class includes utility functions for parsing a serialized MLflow
 * model configuration (`MLModel`) as a {@link Model} object.
 */
public class Model {

  public static class Signature {
    @JsonProperty("inputs")
    private String inputsSchemaJson;
    @JsonProperty("outputs")
    private String outputSchemaJson;
  }

  @JsonProperty("artifact_path")
  private String artifactPath;

  @JsonProperty("run_id")
  private String runId;

  @JsonProperty("utc_time_created")
  private String utcTimeCreated;

  @JsonProperty("flavors")
  private Map<String, Object> flavors;

  @JsonProperty("signature")
  Signature signature;

  @JsonProperty("input_example")
  private Map<String, Object> input_example;

  private String rootPath;

  /**
   * Loads the configuration of an MLflow model and parses it as a {@link Model} object.
   *
   * @param modelRootPath The path to the root directory of the MLflow model
   */
  public static Model fromRootPath(String modelRootPath) throws IOException {
    String configPath = FileUtils.join(modelRootPath, "MLmodel");
    return fromConfigPath(configPath);
  }

  /**
   * Loads the configuration of an MLflow model and parses it as a {@link Model} object.
   *
   * @param configPath The path to the `MLModel` configuration file
   */
  public static Model fromConfigPath(String configPath) throws IOException {
    File configFile = new File(configPath);
    Model model = SerializationUtils.parseYamlFromFile(configFile, Model.class);
    // Set the root path to the directory containing the configuration file.
    // This will be used to create an absolute path to the serialized model
    model.setRootPath(configFile.getParentFile().getAbsolutePath());
    return model;
  }

  /** @return The MLflow model's artifact path */
  public Optional<String> getArtifactPath() {
    return Optional.ofNullable(this.artifactPath);
  }

  /** @return The MLflow model's time of creation */
  public Optional<String> getUtcTimeCreated() {
    return Optional.ofNullable(this.utcTimeCreated);
  }

  /** @return The MLflow model's run id */
  public Optional<String> getRunId() {
    return Optional.ofNullable(this.runId);
  }

  /** @return The path to the root directory of the MLflow model */
  public Optional<String> getRootPath() {
    return Optional.ofNullable(this.rootPath);
  }

  /**
   * Reads the configuration corresponding to the specified flavor name and parses it as a `Flavor`
   * object
   */
  public <T extends Flavor> Optional<T> getFlavor(String flavorName, Class<T> flavorClass) {
    if (this.flavors.containsKey(flavorName)) {
      final ObjectMapper mapper = new ObjectMapper();
      T flavor = mapper.convertValue(this.flavors.get(flavorName), flavorClass);
      return Optional.of(flavor);
    } else {
      return Optional.<T>empty();
    }
  }

  private void setRootPath(String rootPath) {
    this.rootPath = rootPath;
  }
}
