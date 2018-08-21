package org.mlflow.utils;

import java.nio.file.Path;
import java.nio.file.Paths;

/** Utilities for manipulating files and file paths */
public class FileUtils {
  /** Concatenates file paths together and returns the result as a string */
  public static String join(String basePath, String... morePaths) {
    Path filePath = Paths.get(basePath, morePaths);
    return filePath.toString();
  }
}
