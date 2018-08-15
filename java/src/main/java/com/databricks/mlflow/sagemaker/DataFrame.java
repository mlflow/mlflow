package com.databricks.mlflow.sagemaker;

/**
 * Input/output data representation for use by {@link com.databricks.mlflow.sagemaker.Predictor}
 * objects
 */
public class DataFrame {
    enum ContentType { Json, Csv }

    private final String content;
    private final ContentType contentType;

    private DataFrame(String content, ContentType contentType) {
        this.content = content;
        this.contentType = contentType;
    }

    /**
     * Produces a DataFrame from JSON content
     *
     * @param jsonContent A string in valid JSON format
     */
    protected static DataFrame fromJson(String jsonContent) {
        return new DataFrame(jsonContent, ContentType.Json);
    }

    /**
     * Produces a DataFrame from CSV content
     *
     * @param jsonContent A string in valid CSV format
     */
    protected static DataFrame fromCsv(String csvContent) {
        throw new UnsupportedOperationException(
            "Loading dataframes from CSV is not yet supported!");
    }

    /**
     * Produces a JSON string representation of the DataFrame
     *
     * @return A string in JSON format
     */
    protected String toJson() {
        if (this.contentType == ContentType.Json) {
            return this.content;
        } else {
            throw new UnsupportedOperationException(
                "Converting a dataframe of a non-JSON content type to JSON is not yet supported.");
        }
    }

    /**
     * Produces a CSV string representation of the DataFrame
     *
     * @return A string in CSV format
     */
    protected String toCsv() {
        throw new UnsupportedOperationException(
            "Converting a dataframe to CSV is not yet supported.");
    }
}
