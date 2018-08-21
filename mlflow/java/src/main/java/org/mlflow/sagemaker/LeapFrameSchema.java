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
  private static final String LEAP_FRAME_KEY_ROWS = "rows";
  private static final String LEAP_FRAME_KEY_SCHEMA = "schema";

  @JsonIgnoreProperties(ignoreUnknown = true)
  static class SchemaField {
    @JsonProperty("name")
    private String name;
  }

  @JsonProperty("fields")
  private List<SchemaField> fields;

  private String text;
  private List<String> orderedFieldNames;

  /**
   * Loads a leap frame schema from a JSON-formatted file
   *
   * @param filePath The path to the JSON-formatted schema file
   */
  protected static LeapFrameSchema fromPath(String filePath) throws IOException {
    LeapFrameSchema newSchema =
        SerializationUtils.parseJsonFromFile(filePath, LeapFrameSchema.class);
    String schemaText = new String(Files.readAllBytes(Paths.get(filePath)));
    newSchema.setText(schemaText);
    newSchema.setOrderedFieldNames();
    return newSchema;
  }

  /**
   * @return The list of dataframe fields expected by the transformer with this schema, in the order
   *     that these fields are expected to appear
   */
  protected List<String> getOrderedFieldNames() {
    return this.orderedFieldNames;
  }

  /**
   * Converts Pandas dataframe JSON in `record` format to MLeap frame JSON in `row` format using the
   * schema defined by this schema object
   *
   * @throws MissingSchemaFieldException If the supplied pandas dataframe is missing a required
   *     field
   */
  protected String applyToPandasRecordJson(String pandasJson) throws IOException {
    List<String> orderedFieldNames = getOrderedFieldNames();
    List<Map<String, Object>> pandasRecords = SerializationUtils.fromJson(pandasJson, List.class);
    List<List<Object>> mleapRows = new ArrayList<>();
    for (Map<String, Object> record : pandasRecords) {
      List<Object> mleapRow = new ArrayList<>();
      for (String fieldName : orderedFieldNames) {
        if (!record.containsKey(fieldName)) {
          throw new MissingSchemaFieldException(fieldName);
        }
        mleapRow.add(record.get(fieldName));
      }
      mleapRows.add(mleapRow);
    }
    String serializedRows = SerializationUtils.toJson(mleapRows);
    String leapFrameJson =
        String.format(
            "{ \"%s\" : %s, \"%s\" : %s }",
            LEAP_FRAME_KEY_ROWS, serializedRows, LEAP_FRAME_KEY_SCHEMA, this.text);
    return leapFrameJson;
  }

  private void setText(String text) {
    this.text = text;
  }

  private void setOrderedFieldNames() {
    this.orderedFieldNames = new ArrayList<>();
    for (SchemaField field : fields) {
      this.orderedFieldNames.add(field.name);
    }
  }
}
