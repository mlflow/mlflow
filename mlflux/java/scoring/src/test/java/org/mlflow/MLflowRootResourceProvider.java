package org.mlflow;


/** Provides test resources in the org/mlflow test resources directory */
public class MLflowRootResourceProvider {
  /**
   * @param relativePath The path to the requested resource, relative to the `org/mlflow` test
   *     resources directory
   * @return The absolute path to the requested resource
   */
  public static String getResourcePath(String relativePath) {
    return MLflowRootResourceProvider.class.getResource(relativePath).getFile();
  }
}
