package com.databricks.mlflow.sagemaker;

import com.databricks.mlflow.mleap.LeapFrameUtils;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;

public class DataFrame {
    enum ContentType { Json, Csv }

    private final String content;
    private final ContentType contentType;

    private DataFrame(String content, ContentType contentType) {
        this.content = content;
        this.contentType = contentType;
    }

    protected static DataFrame fromJson(String jsonContent) {
        return new DataFrame(jsonContent, ContentType.Json);
    }

    protected static DataFrame fromCsv(String csvContent) {
        throw new UnsupportedOperationException(
            "Loading dataframes from CSV is not yet supported!");
    }

    protected static DataFrame fromLeapFrame(DefaultLeapFrame leapFrame) {
        return fromJson(LeapFrameUtils.getJsonFromLeapFrame(leapFrame));
    }

    protected String toJson() {
        if (this.contentType == ContentType.Json) {
            return this.content;
        } else {
            throw new UnsupportedOperationException(
                "Converting a dataframe of a non-JSON content type to JSON is not yet supported.");
        }
    }

    protected String toCsv() {
        throw new UnsupportedOperationException(
            "Converting a dataframe to CSV is not yet supported.");
    }
}
