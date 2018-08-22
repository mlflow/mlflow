package org.mlflow.utils;

/** Utilities for reading from / writing to the system enviroment */
public interface Environment {
  /** Attempt to parse the value of the specified environment variable as an integer */
  public int getIntegerValue(String varName, int defaultvalue);
}
