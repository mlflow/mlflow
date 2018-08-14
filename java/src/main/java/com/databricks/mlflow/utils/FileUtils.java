package com.databricks.mlflow.utils;

import java.nio.file.Path;
import java.nio.file.Paths;

public class FileUtils {
    public static String join(String basePath, String... morePaths) {
        Path filePath = Paths.get(basePath, morePaths);
        return filePath.toString();
    }
}
