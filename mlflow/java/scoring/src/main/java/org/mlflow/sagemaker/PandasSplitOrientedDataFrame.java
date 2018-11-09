package org.mlflow.sagemaker;

import com.fasterxml.jackson.core.JsonProcessingException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;
import org.mlflow.utils.SerializationUtils;

/**
 * A representation of a serialized Pandas DataFrame in split-oriented format. For more information,
 * see `pandas.DataFrame.toJson(orient="split")`
 * (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_json.html)
 */
class PandasSplitOrientedDataFrame {
  private List<LinkedHashMap<String, Object>> entries;

  private static final String PANDAS_FRAME_KEY_COLUMN_NAMES = "columns";
  private static final String PANDAS_FRAME_KEY_ROWS = "data";

  private static final String LEAP_FRAME_KEY_ROWS = "rows";
  private static final String LEAP_FRAME_KEY_SCHEMA = "schema";

  private PandasSplitOrientedDataFrame(List<String> columnNames, List<List<Object>> rows) {
    this.entries = new ArrayList<>();
    for (int rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
      List<Object> row = rows.get(rowIndex);
      if (row.size() != columnNames.size()) {
        throw new IllegalArgumentException(
            String.format(
                "Row %d of the DataFrame does not contain the expected number of columns! Found %d"
                    + " columns, expected %d columns",
                rowIndex, row.size(), columnNames.size()));
      }
      LinkedHashMap<String, Object> newEntry = new LinkedHashMap<>(row.size());
      for (int i = 0; i < row.size(); ++i) {
        newEntry.put(columnNames.get(i), row.get(i));
      }
      this.entries.add(newEntry);
    }
  }

  /**
   * Constructs a {@link PandasSplitOrientedDataFrame}
   *
   * @param frameJson A representation of the DataFrame
   */
  static PandasSplitOrientedDataFrame fromJson(String frameJson) throws IOException {
    Map<String, List<?>> parsedFrame = SerializationUtils.fromJson(frameJson, Map.class);
    validatePandasDataFrameJsonRepresentation(parsedFrame);
    return new PandasSplitOrientedDataFrame(
        (List<String>) parsedFrame.get(PANDAS_FRAME_KEY_COLUMN_NAMES),
        (List<List<Object>>) parsedFrame.get(PANDAS_FRAME_KEY_ROWS));
  }

  private static void validatePandasDataFrameJsonRepresentation(Map<String, List<?>> parsedFrame)
      throws InvalidSchemaException {
    String[] expectedColumnNames =
        new String[] {PANDAS_FRAME_KEY_COLUMN_NAMES, PANDAS_FRAME_KEY_ROWS};
    for (String columnName : expectedColumnNames) {
      if (!parsedFrame.containsKey(columnName)) {
        throw new InvalidSchemaException(
            String.format(
                "The JSON representation of the serialized Pandas DataFrame is missing an expected "
                    + " column with name: `%s` that is required by the Pandas `split` orientation.",
                columnName));
      }
    }
  }

  /** @return The number of rows contained in the DataFrame */
  int size() {
    return this.entries.size();
  }

  /**
   * Applies the specified MLeap frame schema ({@link LeapFrameSchema}) to this DataFrame, producing
   * a {@link DefaultLeapFrame}
   *
   * @throws InvalidSchemaException If the supplied pandas DataFrame is invalid (missing a required
   *     field, etc)
   */
  DefaultLeapFrame toLeapFrame(LeapFrameSchema leapFrameSchema) throws JsonProcessingException {
    List<List<Object>> mleapRows = new ArrayList<>();
    for (Map<String, Object> entry : this.entries) {
      List<Object> mleapRow = new ArrayList<>();
      for (String fieldName : leapFrameSchema.getFieldNames()) {
        if (!entry.containsKey(fieldName)) {
          throw new InvalidSchemaException(
              String.format("Pandas DataFrame is missing a required field: `%s`", fieldName));
        }
        mleapRow.add(entry.get(fieldName));
      }
      mleapRows.add(mleapRow);
    }
    Map<String, Object> rawFrame = new HashMap<>();
    rawFrame.put(LEAP_FRAME_KEY_ROWS, mleapRows);
    rawFrame.put(LEAP_FRAME_KEY_SCHEMA, leapFrameSchema.getRawSchema());
    String leapFrameJson = SerializationUtils.toJson(rawFrame);
    DefaultLeapFrame leapFrame = LeapFrameUtils.getLeapFrameFromJson(leapFrameJson);
    return leapFrame;
  }
}
