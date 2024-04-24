package org.mlflow.tracking.utils;

public class MlflowTagConstants {
  public static final String PARENT_RUN_ID = "mlflow.parentRunId";
  public static final String RUN_NAME = "mlflow.runName";
  public static final String USER = "mlflow.user";
  public static final String SOURCE_TYPE = "mlflow.source.type";
  public static final String SOURCE_NAME = "mlflow.source.name";
  public static final String DATABRICKS_NOTEBOOK_ID = "mlflow.databricks.notebookID";
  public static final String DATABRICKS_NOTEBOOK_PATH = "mlflow.databricks.notebookPath";
  // The JOB_ID, JOB_RUN_ID, and JOB_TYPE tags are used for automatically recording Job 
  // information when MLflow Tracking APIs are used within a Databricks Job
  public static final String DATABRICKS_JOB_ID = "mlflow.databricks.jobID";
  public static final String DATABRICKS_JOB_RUN_ID = "mlflow.databricks.jobRunID";
  public static final String DATABRICKS_JOB_TYPE = "mlflow.databricks.jobType";
  public static final String DATABRICKS_WEBAPP_URL = "mlflow.databricks.webappURL";
  public static final String MLFLOW_EXPERIMENT_SOURCE_TYPE = "mlflow.experiment.sourceType";
  public static final String MLFLOW_EXPERIMENT_SOURCE_ID = "mlflow.experiment.sourceId";
}
