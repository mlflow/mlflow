package org.mlflow.sagemaker;

import com.fasterxml.jackson.core.JsonProcessingException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;
import org.mlflow.utils.SerializationUtils;

/**
 * A representation of a serialized Pandas dataframe in record-oriented format. For more
 * information, see `pandas.DataFrame.toJson(orient="records")`
 * (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_json.html)
 */
class PandasRecordOrientedDataFrame {
  private final List<Map<String, Object>> records;

  private static final String LEAP_FRAME_KEY_ROWS = "rows";
  private static final String LEAP_FRAME_KEY_SCHEMA = "schema";

  private PandasRecordOrientedDataFrame(List<Map<String, Object>> records) {
    this.records = records;
  }

  /**
   * Constructs a {@link PandasRecordOrientedDataFrame}
   *
   * @param frameJson A representation of the dataframe
   */
  static PandasRecordOrientedDataFrame fromJson(String frameJson) throws IOException {
    return new PandasRecordOrientedDataFrame(SerializationUtils.fromJson(frameJson, List.class));
  }

  /** @return The number of records contained in the dataframe */
  int size() {
    return this.records.size();
  }

  /**
   * Applies the specified MLeap frame schema ({@link LeapFrameSchema}) to this dataframe, producing
   * a {@link DefaultLeapFrame}
   *
   * @throws InvalidSchemaException If the supplied pandas dataframe is invalid (missing a required
   *     field, etc)
   */
  DefaultLeapFrame toLeapFrame(LeapFrameSchema leapFrameSchema) throws JsonProcessingException {
    List<List<Object>> mleapRows = new ArrayList<>();
    for (Map<String, Object> record : this.records) {
      List<Object> mleapRow = new ArrayList<>();
      for (String fieldName : leapFrameSchema.getFieldNames()) {
        if (!record.containsKey(fieldName)) {
          throw new InvalidSchemaException(
              String.format("Pandas dataframe is missing a required field: `%s`", fieldName));
        }
        mleapRow.add(record.get(fieldName));
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
