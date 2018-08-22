package org.mlflow.sagemaker;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.mlflow.utils.SerializationUtils;

/** Representation of the dataframe schema that an {@link MLeapPredictor} expects inputs to have */
@JsonIgnoreProperties(ignoreUnknown = true)
class LeapFrameSchema {
  private final Map<String, Object> rawSchema;
  private final List<String> fields;

  private LeapFrameSchema(Map<String, Object> rawSchema) {
    this.rawSchema = rawSchema;
    if (!rawSchema.containsKey("fields")) {
      throw new InvalidSchemaException("Leap frame schema must contain a top-level `fields` key!");
    }
    this.fields = new ArrayList<String>();
    for(
  }

  public static class InvalidSchemaException extends RuntimeException {
    InvalidSchemaException(String message) {
      super(message);
    }
  }

  static LeapFrameSchema fromPath(String filePath) {}

  // @JsonIgnoreProperties(ignoreUnknown = true)
  // static class SchemaField {
  //   @JsonProperty("name") private String name;
  // }
  //
  // @JsonProperty("fields") private List<SchemaField> fields;
  //
  // private String schemaText;
  // private List<String> fieldNames;
  //
  // /**
  //  * Loads a leap frame schema from a JSON-formatted file
  //  *
  //  * @param filePath The path to the JSON-formatted schema file
  //  */
  // protected static LeapFrameSchema fromPath(String filePath) throws IOException {
  //   LeapFrameSchema newSchema =
  //       SerializationUtils.parseJsonFromFile(filePath, LeapFrameSchema.class);
  //   String schemaText = new String(Files.readAllBytes(Paths.get(filePath)));
  //   newSchema.setSchemaText(schemaText);
  //   newSchema.setFieldNames();
  //   return newSchema;
  // }
  //
  // /**
  //  * @return The list of dataframe fields expected by the transformer with this schema, in the
  //  order
  //  *     that these fields are expected to appear
  //  */
  // List<String> getFieldNames() {
  //   return this.fieldNames;
  // }
  //
  // String getSchemaText() {
  //   return this.schemaText;
  // }
  //
  // private void setSchemaText(String schemaText) {
  //   this.schemaText = schemaText;
  // }
  //
  // private void setFieldNames() {
  //   this.fieldNames = new ArrayList<>();
  //   for (SchemaField field : fields) {
  //     this.fieldNames.add(field.name);
  //   }
  // }
}
