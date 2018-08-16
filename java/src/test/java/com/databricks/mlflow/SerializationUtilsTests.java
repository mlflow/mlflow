package com.databricks.mlflow.utils;

import org.junit.Test;
import org.junit.Assert;

import java.io.IOException;
import java.io.File;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

/**
 * Unit tests for the {@link com.databricks.mlflow.utils.SerializationUtils} module
 */
public class SerializationUtilsTests {
    @Test
    public void testSupportedJavaObjectsCanBeJsonSerializedAndDeserializedInTheSameFormat() {
        List<Integer> inputList = new ArrayList<>();
        for (int i = 0; i < 100; ++i) {
            inputList.add(i);
        }
        try {
            String listJson = SerializationUtils.toJson(inputList);
            List<Integer> loadedList = SerializationUtils.fromJson(listJson);
            Assert.assertEquals(loadedList, inputList);
        } catch (IOException e) {
            Assert.fail(
                "Encountered an exception while serializing and deserializing list JSON content!");
        }

        Map<String, Integer> inputMap = new HashMap<>();
        for (int i = 0; i < 100; ++i) {
            inputMap.put(String.valueOf(i), i);
        }

        try {
            String mapJson = SerializationUtils.toJson(inputMap);
            Map<String, Integer> loadedMap = SerializationUtils.fromJson(mapJson);
            Assert.assertEquals(loadedMap, inputMap);
        } catch (IOException e) {
            Assert.fail(
                "Encountered an exception while serializing and deserializing map JSON content!");
        }
    }

    @Test
    public void testJsonObjectsAreParsedFromFilesAndFilePathsCorrectly() throws IOException {
        Map<String, Object> inputMap = new HashMap<>();
        inputMap.put("field1", 1);
        inputMap.put("field2", "two");
        inputMap.put("field3", 3.0);

        File tempFile = File.createTempFile("json", "filetest", new File("/tmp"));
        String tempFilePath = tempFile.getAbsolutePath();
        tempFile.deleteOnExit();

        ObjectMapper jsonMapper = new ObjectMapper(new JsonFactory());
        jsonMapper.writeValue(tempFile, inputMap);

        Map<String, Object> loadedFileMap =
            SerializationUtils.parseJsonFromFile(tempFile, Map.class);
        Assert.assertEquals(inputMap, loadedFileMap);

        Map<String, Object> loadedPathMap =
            SerializationUtils.parseJsonFromFile(tempFilePath, Map.class);
        Assert.assertEquals(inputMap, loadedPathMap);
    }

    @Test
    public void testYamlObjectsAreParsedFromFilesAndFilePathsCorrectly() throws IOException {
        Map<String, Object> inputMap = new HashMap<>();
        inputMap.put("field1", 1);
        inputMap.put("field2", "two");
        inputMap.put("field3", 3.0);

        File tempFile = File.createTempFile("yaml", "filetest", new File("/tmp"));
        String tempFilePath = tempFile.getAbsolutePath();
        tempFile.deleteOnExit();

        ObjectMapper jsonMapper = new ObjectMapper(new YAMLFactory());
        jsonMapper.writeValue(tempFile, inputMap);

        Map<String, Object> loadedFileMap =
            SerializationUtils.parseYamlFromFile(tempFile, Map.class);
        Assert.assertEquals(inputMap, loadedFileMap);

        Map<String, Object> loadedPathMap =
            SerializationUtils.parseYamlFromFile(tempFilePath, Map.class);
        Assert.assertEquals(inputMap, loadedPathMap);
    }
}
