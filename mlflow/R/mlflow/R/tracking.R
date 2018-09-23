#' Create Experiment
#'
#' Creates an MLflow experiment.
#'
#' @param name The name of the experiment to create.
#' @param artifact_location Location where all artifacts for this experiment are stored. If
#'   not provided, the remote server will select an appropriate default.
#'
#' @export
mlflow_create_experiment <- function(name, artifact_location = NULL, client = NULL) {
  UseMethod("mlflow_create_experiment", client)
}

#' @export
mlflow_create_experiment.mlflow_client <- function(name, artifact_location = NULL, client = NULL) {
  name <- forge::cast_string(name)
  experiment_id <- mlflow_client_create_experiment(client, name, artifact_location)
  invisible(experiment_id)
}

#' Create Run
#'
#' reate a new run within an experiment. A run is usually a single execution of a machine learning or data ETL pipeline.
#'
#' MLflow uses runs to track Param, Metric, and RunTag, associated with a single execution.
#'
#' @param experiment_id Unique identifier for the associated experiment.
#' @param user_id User ID or LDAP for the user executing the run.
#' @param run_name Human readable name for run.
#' @param source_type Originating source for this run. One of Notebook, Job, Project, Local or Unknown.
#' @param source_name String descriptor for source. For example, name or description of the notebook, or job name.
#' @param status Current status of the run. One of RUNNING, SCHEDULE, FINISHED, FAILED, KILLED.
#' @param start_time Unix timestamp of when the run started in milliseconds.
#' @param end_time Unix timestamp of when the run ended in milliseconds.
#' @param source_version Git version of the source code used to create run.
#' @param entry_point_name Name of the entry point for the run.
#' @param tags Additional metadata for run in key-value pairs.
#' @export
mlflow_create_run <- function(
  experiment_id, user_id = NULL, run_name = NULL, source_type = NULL,
  source_name = NULL, entry_point_name = NULL, start_time = NULL, source_version = NULL,
  tags = NULL, client = NULL
) {
  UseMethod("mlflow_create_run", client)
}

#' @export
mlflow_create_run.mlflow_client <- function(
  experiment_id, user_id = NULL, run_name = NULL, source_type = NULL,
  source_name = NULL, entry_point_name = NULL, start_time = NULL, source_version = NULL,
  tags = NULL, client = NULL
) {
  tags <- if (!is.null(tags)) tags %>%
    purrr::imap(~ list(key = .y, value = .x)) %>%
    unname()

  mlflow_client_create_run(
    client, experiment_id, user_id, run_name, source_type,
    source_name, entry_point_name, start_time, source_version, tags
  )
}

#' Delete Experiment
#'
#' Mark an experiment and associated runs, params, metrics, … etc for deletion. If the
#'   experiment uses FileStore, artifacts associated with experiment are also deleted.
#'
#' @param experiment_id ID of the associated experiment This field is required.
#' @export
mlflow_delete_experiment <- function(experiment_id, client = NULL) {
  UseMethod("mlflow_delete_experiment", client)
}

#' @export
mlflow_delete_experiment.mlflow_client <- function(experiment_id, client = NULL) {
  mlflow_client_delete_experiment(client, experiment_id)
}

#' Restore Experiment
#'
#' Restore an experiment marked for deletion. This also restores associated metadata,
#'   runs, metrics, and params. If experiment uses FileStore, underlying artifacts
#'   associated with experiment are also restored.
#'
#' Throws RESOURCE_DOES_NOT_EXIST if experiment was never created or was permanently deleted.
#'
#' @param experiment_id ID of the associated experiment This field is required.
#' @export
mlflow_restore_experiment <- function(experiment_id, client = NULL) {
  UseMethod("mlflow_restore_experiment", client)
}

#' @export
mlflow_restore_experiment.mlflow_client <- function(experiment_id, client = NULL) {
  mlflow_client_restore_experiment(client, experiment_id)
}
