package com.databricks.mlflow.utils;

import java.io.File;
import java.io.IOException;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

public class SerializationUtils {
    private static final ObjectMapper jsonMapper = new ObjectMapper(new JsonFactory());
    private static final ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());

    public static String toJson(Object object) throws JsonProcessingException {
        return jsonMapper.writeValueAsString(object);
    }

    public static <T> T fromJson(String json) throws IOException {
        return jsonMapper.readValue(json, new TypeReference<T>() {});
    }

    public static <T> T parseJsonFromFile(String filePath, Class<T> objectClass)
        throws IOException {
        File jsonFile = new File(filePath);
        return parseJsonFromFile(jsonFile, objectClass);
    }

    public static <T> T parseJsonFromFile(File jsonFile, Class<T> objectClass) throws IOException {
        return parseFromFile(jsonFile, objectClass, jsonMapper);
    }

    public static <T> T parseYamlFromFile(String filePath, Class<T> objectClass)
        throws IOException {
        File yamlFile = new File(filePath);
        return parseYamlFromFile(yamlFile, objectClass);
    }

    public static <T> T parseYamlFromFile(File yamlFile, Class<T> objectClass) throws IOException {
        return parseFromFile(yamlFile, objectClass, yamlMapper);
    }

    private static <T> T parseFromFile(File file, Class<T> objectClass, ObjectMapper mapper)
        throws IOException {
        T parsedObject = mapper.readValue(file, objectClass);
        return parsedObject;
    }
}
