package org.mlflow;

/** Interface for exposing information about an MLFlow model flavor. */
public interface Flavor {
  /** @return The name of the model flavor */
  String getName();

  /**
   * @return The relative path to flavor-specific model data. This path is relative to the root
   *     directory of an MLFlow model
   */
  String getModelDataPath();
}
