package com.databricks.mlflow.utils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

public class FileUtils {
    public static String join(String basePath, String... morePaths) {
        Path filePath = Paths.get(basePath, morePaths);
        return filePath.toString();
    }

    public static <T> T parseJsonFromFile(String filePath, Class<T> objectClass)
        throws IOException {
        File jsonFile = new File(filePath);
        return parseJsonFromFile(jsonFile, objectClass);
    }

    public static <T> T parseJsonFromFile(File jsonFile, Class<T> objectClass) throws IOException {
        final ObjectMapper jsonMapper = new ObjectMapper(new JsonFactory());
        return parseFromFile(jsonFile, objectClass, jsonMapper);
    }

    public static <T> T parseYamlFromFile(String filePath, Class<T> objectClass)
        throws IOException {
        File yamlFile = new File(filePath);
        return parseYamlFromFile(yamlFile, objectClass);
    }

    public static <T> T parseYamlFromFile(File yamlFile, Class<T> objectClass) throws IOException {
        final ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());
        return parseFromFile(yamlFile, objectClass, yamlMapper);
    }

    private static <T> T parseFromFile(File file, Class<T> objectClass, ObjectMapper mapper)
        throws IOException {
        T parsedObject = mapper.readValue(file, objectClass);
        return parsedObject;
    }
}
