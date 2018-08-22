package org.mlflow.utils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.commons.io.IOUtils;

/** Utilities for manipulating files and file paths */
public class FileUtils {
  /** Concatenates file paths together and returns the result as a string */
  public static String join(String basePath, String... morePaths) {
    Path filePath = Paths.get(basePath, morePaths);
    return filePath.toString();
  }

  /** Reads an input stream as a UTF-8 encoded string */
  public static String readInputStreamAsUtf8(InputStream inputStream) throws IOException {
    return IOUtils.toString(inputStream, StandardCharsets.UTF_8);
  }
}
