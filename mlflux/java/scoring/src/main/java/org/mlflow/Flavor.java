package org.mlflow;

/** Interface for exposing information about an MLflow model flavor. */
public interface Flavor {
  /** @return The name of the model flavor */
  String getName();

  /**
   * @return The relative path to flavor-specific model data. This path is relative to the root
   *     directory of an MLflow model
   */
  String getModelDataPath();
}
