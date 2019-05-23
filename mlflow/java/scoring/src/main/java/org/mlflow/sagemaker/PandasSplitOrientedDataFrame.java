package org.mlflow.sagemaker;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ml.combust.mleap.core.types.StructField;
import ml.combust.mleap.core.types.StructType;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;
import ml.combust.mleap.runtime.frame.Row;
import ml.combust.mleap.runtime.javadsl.LeapFrameBuilder;
import ml.combust.mleap.runtime.javadsl.LeapFrameSupport;
import org.mlflow.utils.SerializationUtils;

/**
 * A representation of a serialized Pandas DataFrame in split-oriented format. For more information,
 * see `pandas.DataFrame.toJson(orient="split")`
 * (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_json.html)
 */
class PandasSplitOrientedDataFrame {

  private static final LeapFrameBuilder leapFrameBuilder = new LeapFrameBuilder();
  private static final LeapFrameSupport leapFrameSupport = new LeapFrameSupport();

  private final List<LinkedHashMap<String, Object>> entries;

  private static final String PANDAS_FRAME_KEY_COLUMN_NAMES = "columns";
  private static final String PANDAS_FRAME_KEY_ROWS = "data";

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
   * Applies the specified MLeap frame schema ({@link StructType}) to this DataFrame, producing
   * a {@link DefaultLeapFrame}
   *
   * @throws InvalidSchemaException If the supplied pandas DataFrame is invalid (missing a required
   *     field, etc)
   */
  DefaultLeapFrame toLeapFrame(StructType leapFrameSchema) {
    List<Row> mleapRows = new ArrayList<>();

    for (Map<String, Object> entry : this.entries) {
      List<Object> mleapRow = new ArrayList<>();
      for (StructField field : leapFrameSupport.getFields(leapFrameSchema)) {
        String fieldName = field.name();
        if (!entry.containsKey(fieldName)) {
          throw new InvalidSchemaException(
              String.format("Pandas DataFrame is missing a required field: `%s`", fieldName));
        }
        mleapRow.add(entry.get(fieldName));
      }
      mleapRows.add(leapFrameBuilder.createRowFromIterable(mleapRow));
    }

    return leapFrameBuilder.createFrame(leapFrameSchema, mleapRows);
  }
}
