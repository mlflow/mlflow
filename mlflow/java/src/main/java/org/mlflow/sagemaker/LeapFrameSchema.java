package org.mlflow.sagemaker;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.mlflow.utils.SerializationUtils;

/** Representation of the dataframe schema that an {@link MLeapPredictor} expects inputs to have */
class LeapFrameSchema {
  private final Map<String, Object> rawSchema;
  private final List<String> fieldNames;

  @JsonIgnoreProperties(ignoreUnknown = true)
  private static class SchemaField {
    @JsonProperty("name") String name;
  }

  private LeapFrameSchema(Map<String, Object> rawSchema) {
    this.rawSchema = rawSchema;
    if (!rawSchema.containsKey("fields")) {
      throw new InvalidSchemaException("Leap frame schema must contain a top-level `fields` key!");
    }

    final ObjectMapper mapper = new ObjectMapper();
    List<SchemaField> fields =
        mapper.convertValue(rawSchema.get("fields"), new TypeReference<List<SchemaField>>() {});

    this.fieldNames = new ArrayList<String>();
    for (SchemaField field : fields) {
      fieldNames.add(field.name);
    }
  }

  static LeapFrameSchema fromPath(String filePath) throws IOException {
    return new LeapFrameSchema(
        (Map<String, Object>) SerializationUtils.parseJsonFromFile(filePath, Map.class));
  }

  public static class InvalidSchemaException extends RuntimeException {
    InvalidSchemaException(String message) {
      super(message);
    }
  }

  /**
   * @return The list of dataframe fields expected by the transformer with this schema, in the
   order
   *     that these fields are expected to appear
   */
  List<String> getFieldNames() {
    return this.fieldNames;
  }

  Map<String, Object> getRawSchema() {
    return this.rawSchema;
  }
}
