package org.mlflow.utils;

/**
 * Interface defining functions that should be implemented by an environment consisting of keys and
 * values
 */
public interface Environment {
  /** Attempt to parse the value of the specified environment variable as an integer */
  public int getIntegerValue(String varName, int defaultvalue);
}
