package org.mlflow.sagemaker;

import java.nio.file.Files;
import java.nio.file.Paths;
import org.junit.Assert;
import org.junit.Test;
import java.io.File;
import org.mlflow.utils.SerializationUtils;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.mlflow.MLflowRootResourceProvider;

import ml.combust.mleap.runtime.frame.DefaultLeapFrame;
import com.fasterxml.jackson.core.JsonProcessingException;

public class PandasDataFrameTest {
  @Test
  public void testPandasDataFrameIsProducedFromValidJsonSuccessfully() throws IOException {
    String sampleInputPath =
        MLflowRootResourceProvider.getResourcePath("mleap_model/sample_input.json");
    String sampleInputJson = new String(Files.readAllBytes(Paths.get(sampleInputPath)));
    PandasSplitOrientedDataFrame pandasFrame =
        PandasSplitOrientedDataFrame.fromJson(sampleInputJson);
    Assert.assertEquals((pandasFrame.size() == 1), true);
  }

  @Test
  public void testLoadingPandasDataFrameFromInvalidJsonThrowsIOException() {
    String badFrameJson = "this is not valid frame json";
    try {
      PandasSplitOrientedDataFrame pandasFrame =
          PandasSplitOrientedDataFrame.fromJson(badFrameJson);
      Assert.fail("Expected parsing a pandas dataframe from invalid json to throw an IOException.");
    } catch (IOException e) {
      // Succeed
    }
  }

  @Test
  public void testLoadingPandasDataFrameFromJsonWithInvalidSplitOrientationSchemaThrowsException()
      throws IOException, JsonProcessingException {
    String sampleInputPath =
        MLflowRootResourceProvider.getResourcePath("mleap_model/sample_input.json");
    String sampleInputJson = new String(Files.readAllBytes(Paths.get(sampleInputPath)));
    Map<String, List<?>> sampleInput = SerializationUtils.fromJson(sampleInputJson, Map.class);
    sampleInput.remove("columns");
    String missingSchemaFieldJson = SerializationUtils.toJson(sampleInput);

    try {
      PandasSplitOrientedDataFrame pandasFrame =
          PandasSplitOrientedDataFrame.fromJson(missingSchemaFieldJson);
      Assert.fail("Expected parsing a pandas dataframe with an invalid `split` orientation schema"
          + " to throw an exception.");
    } catch (InvalidSchemaException e) {
      // Succeed
    }
  }

  @Test
  public void testPandasDataFrameWithMLeapCompatibleSchemaIsConvertedToLeapFrameSuccessfully()
      throws JsonProcessingException, IOException {
    String schemaPath = MLflowRootResourceProvider.getResourcePath("mleap_model/mleap/schema.json");
    LeapFrameSchema leapFrameSchema = LeapFrameSchema.fromPath(schemaPath);

    String sampleInputPath =
        MLflowRootResourceProvider.getResourcePath("mleap_model/sample_input.json");
    String sampleInputJson = new String(Files.readAllBytes(Paths.get(sampleInputPath)));
    PandasSplitOrientedDataFrame pandasFrame =
        PandasSplitOrientedDataFrame.fromJson(sampleInputJson);

    DefaultLeapFrame leapFrame = pandasFrame.toLeapFrame(leapFrameSchema);
  }

  /**
   * In order to produce a leap frame from a pandas dataframe, the pandas dataframe
   * must contain all of the fields specified by the intended leap frame's schema.
   * This test ensures that an exception is thrown if such a field is missing
   */
  @Test
  public void testConvertingPandasDataFrameWithMissingMLeapSchemaFieldThrowsException()
      throws IOException, JsonProcessingException {
    String schemaPath = MLflowRootResourceProvider.getResourcePath("mleap_model/mleap/schema.json");
    LeapFrameSchema leapFrameSchema = LeapFrameSchema.fromPath(schemaPath);

    String sampleInputPath =
        MLflowRootResourceProvider.getResourcePath("mleap_model/sample_input.json");
    String sampleInputJson = new String(Files.readAllBytes(Paths.get(sampleInputPath)));
    Map<String, List<?>> sampleInput = SerializationUtils.fromJson(sampleInputJson, Map.class);
    List<List<Object>> rows = (List<List<Object>>) sampleInput.get("data");
    List<String> columnNames = (List<String>) sampleInput.get("columns");
    int topicIndex = columnNames.indexOf("topic");
    columnNames.remove("topic");
    for (List<Object> row : rows) {
      row.remove(topicIndex);
    }
    String missingDataColumnJson = SerializationUtils.toJson(sampleInput);

    PandasSplitOrientedDataFrame pandasFrame =
        PandasSplitOrientedDataFrame.fromJson(missingDataColumnJson);
    try {
      pandasFrame.toLeapFrame(leapFrameSchema);
      Assert.fail(
          "Expected leap frame conversion of a pandas dataframe with a missing field to fail.");
    } catch (InvalidSchemaException e) {
      // Succeed
    }
  }
}
