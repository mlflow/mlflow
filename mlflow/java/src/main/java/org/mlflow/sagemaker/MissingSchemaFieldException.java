package org.mlflow.sagemaker;

public class MissingSchemaFieldException extends RuntimeException {
  private final String fieldName;

  protected MissingSchemaFieldException(String fieldName) {
    this.fieldName = fieldName;
  }

  public String getMissingFieldName() {
    return this.fieldName;
  }
}
