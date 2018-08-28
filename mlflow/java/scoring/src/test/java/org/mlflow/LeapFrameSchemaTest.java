package org.mlflow.sagemaker;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;
import org.mlflow.MLflowRootResourceProvider;
import org.mlflow.utils.SerializationUtils;

/** Unit tests for the {@link LeapFrameSchema} module */
public class LeapFrameSchemaTest {
  @Test
  public void testLeapFrameSchemaIsLoadedFromValidPathWithCorrectFieldOrder() throws IOException {
    String schemaPath = MLflowRootResourceProvider.getResourcePath("mleap_model/mleap/schema.json");
    LeapFrameSchema schema = LeapFrameSchema.fromPath(schemaPath);
    List<String> orderedFieldNames = schema.getFieldNames();
    List<String> expectedOrderedFieldNames = Arrays.asList("text", "topic");
    Assert.assertEquals(orderedFieldNames, expectedOrderedFieldNames);
  }
}
