package org.mlflow.mleap;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.mlflow.Flavor;

/** Represents an MLeap flavor configuration */
public class MLeapFlavor implements Flavor {
  public static final String FLAVOR_NAME = "mleap";

  @JsonProperty("mleap_version")
  private String mleapVersion;

  @JsonProperty("model_data")
  private String modelDataPath;

  @JsonProperty("training_schema")
  private String trainingSchemaPath;

  @Override
  public String getName() {
    return FLAVOR_NAME;
  }

  @Override
  public String getModelDataPath() {
    return modelDataPath;
  }

  /** @return The version of MLeap with which the mode data was serialized */
  public String getMleapVersion() {
    return mleapVersion;
  }

  /**
   * @return The relative path to the model's serialized training schema. This path is relative to
   *     the root directory of an MLFlow model
   */
  public String getTrainingSchemaPath() {
    return trainingSchemaPath;
  }
}
