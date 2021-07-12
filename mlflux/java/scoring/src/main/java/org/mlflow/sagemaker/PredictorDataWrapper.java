package org.mlflow.sagemaker;

/** Input/output data representation for use by {@link org.mlflow.sagemaker.Predictor} objects */
public class PredictorDataWrapper {
  public enum ContentType {
    Json,
    Csv
  }

  private final String content;
  private final ContentType contentType;

  /** Constructs a PredictorDataWrapper */
  public PredictorDataWrapper(String content, ContentType contentType) {
    this.content = content;
    this.contentType = contentType;
  }

  /** @return The type of content contained in the wrapper (JSON, CSV, etc.) */
  ContentType getContentType() {
    return this.contentType;
  }

  /**
   * Produces a JSON string representation of the PredictorDataWrapper
   *
   * @return A string in JSON format
   */
  String toJson() {
    if (this.contentType == ContentType.Json) {
      return this.content;
    } else {
      throw new UnsupportedOperationException(
          "Converting a data wrapper of a non-JSON content type to JSON is not yet supported.");
    }
  }

  /**
   * Produces a CSV string representation of the PredictorDataWrapper
   *
   * @return A string in CSV format
   */
  String toCsv() {
    throw new UnsupportedOperationException(
        "Converting a data wrapper to CSV is not yet supported.");
  }
}
