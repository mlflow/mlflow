package org.mlflow.utils;

/** Utilities for reading from / writing to the system enviroment */
public class SystemEnvironment implements Environment {
  private static final SystemEnvironment systemEnvironment = new SystemEnvironment();

  private SystemEnvironment() {}

  /** Obtains the system environment */
  public static SystemEnvironment get() {
    return systemEnvironment;
  }

  @Override
  public int getIntegerValue(String varName, int defaultValue) {
    String rawValue = System.getenv(varName);
    if (rawValue == null) {
      return defaultValue;
    } else {
      return Integer.valueOf(rawValue);
    }
  }
}
