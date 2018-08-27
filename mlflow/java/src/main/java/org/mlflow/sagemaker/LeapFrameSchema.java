package org.mlflow.sagemaker;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
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
    @JsonProperty("name")
    String name;
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

  /**
   * @throws InvalidSchemaException If the schema cannot be parsed from JSON or does not contain
   *     required keys
   * @throws IOException If the schema file cannot be loaded from the specified path
   */
  static LeapFrameSchema fromPath(String filePath) throws IOException {
    File schemaFile = new File(filePath);
    try {
      return new LeapFrameSchema(
          (Map<String, Object>) SerializationUtils.parseJsonFromFile(filePath, Map.class));
    } catch (IOException e) {
      throw new InvalidSchemaException("The specified schema could not be parsed as JSON.");
    }
  }

  /**
   * @return The list of dataframe fields expected by the transformer with this schema, in the order
   *     that these fields are expected to appear
   */
  List<String> getFieldNames() {
    return this.fieldNames;
  }

  /**
   * @return A representation of the schema as a map containg standard Java objects. This is useful
   *     for serializing the schema as JSON
   */
  Map<String, Object> getRawSchema() {
    return this.rawSchema;
  }
}
