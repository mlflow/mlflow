package org.mlflow.sagemaker;

import org.junit.Assert;
import org.junit.Test;

import org.mlflow.LoaderModuleTest;
import org.mlflow.utils.SerializationUtils;

import java.io.IOException;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Unit tests for the {@link LeapFrameSchema} module
 */
public class LeapFrameSchemaTest {
  @Test
  public void testLeapFrameSchemaIsLoadedFromValidPathWithCorrectFieldOrder() throws IOException {
    String schemaPath =
        LoaderModuleTest.class.getResource("mleap_model/mleap/schema.json").getFile();
    LeapFrameSchema schema = LeapFrameSchema.fromPath(schemaPath);
    List<String> orderedFieldNames = schema.getOrderedFieldNames();
    List<String> expectedOrderedFieldNames = Arrays.asList("text", "topic");
    Assert.assertEquals(orderedFieldNames, expectedOrderedFieldNames);
  }

  @Test
  public void testLeapFrameSchemaAddsSchemaAndRowKeysToValidPandasInputJson() throws IOException {
    String schemaPath =
        LoaderModuleTest.class.getResource("mleap_model/mleap/schema.json").getFile();
    LeapFrameSchema schema = LeapFrameSchema.fromPath(schemaPath);

    String sampleInputPath =
        LoaderModuleTest.class.getResource("mleap_model/sample_input.json").getFile();
    String sampleInputJson = new String(Files.readAllBytes(Paths.get(sampleInputPath)));
    DataFrame inputDataFrame = DataFrame.fromJson(sampleInputJson);

    String schemaModifiedJson = schema.applyToPandasRecordJson(sampleInputJson);
    Map<String, Object> schemaModifiedObject =
        SerializationUtils.fromJson(schemaModifiedJson, Map.class);
    Assert.assertEquals(schemaModifiedObject.containsKey("schema"), true);
    Assert.assertEquals(schemaModifiedObject.containsKey("rows"), true);
  }

  @Test
  public void
  testLeapFrameSchemaThrowsMissingFieldExceptionWhenAppliedToPandasJsonWithMissingField()
      throws IOException {
    String schemaPath =
        LoaderModuleTest.class.getResource("mleap_model/mleap/schema.json").getFile();
    LeapFrameSchema schema = LeapFrameSchema.fromPath(schemaPath);

    String sampleInputPath =
        LoaderModuleTest.class.getResource("mleap_model/sample_input.json").getFile();
    String sampleInputJson = new String(Files.readAllBytes(Paths.get(sampleInputPath)));
    List<Map<String, Object>> sampleInput =
        SerializationUtils.fromJson(sampleInputJson, List.class);

    sampleInput.get(0).remove("topic");
    String badInputJson = SerializationUtils.toJson(sampleInput);
    try {
      schema.applyToPandasRecordJson(badInputJson);
      Assert.fail("Expected schema application to input frame with missing key to fail.");
    } catch (MissingSchemaFieldException e) {
      // Succeed
    }
  }
}
