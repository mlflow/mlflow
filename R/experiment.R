#' List Experiments
#'
#' Retrieves MLflow experiments as a data frame.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' # list local experiments
#' mlflow_list_experiments()
#'
#' # list experiments in remote MLflow server
#' mlflow_tracking_url("http://tracking-server:5000")
#' mlflow_list_experiments()
#' }
#'
#' @export
mlflow_list_experiments <- function() {
  response <- mlflow_rest("experiments", "list")
  exps <- response$experiments

  exps$artifact_location <- mlflow_relative_paths(exps$artifact_location)
  exps
}

#' Create Experiment
#'
#' Creates an MLflow experiment.
#'
#' @param name The name of the experiment to create.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' # create local experiment
#' mlflow_create_experiment("My Experiment")
#'
#' # create experiment in remote MLflow server
#' mlflow_tracking_url("http://tracking-server:5000")
#' mlflow_experiments_create("My Experiment")
#' }
#'
#' @export
mlflow_create_experiment <- function(name) {
  response <- mlflow_rest("experiments", "create", verb = "POST", data = list(name = name))
  response$experimentId
}

#' Get Experiment
#'
#' Get meta data for experiment and a list of runs for this experiment.
#'
#' @param experiment_id Identifer to get an experiment.
#' @export
mlflow_get_experiment <- function(experiment_id) {
  response <- mlflow_rest("experiments", "get", query = list(experiment_id = experiment_id))
  response
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
#' @param artifact_uri URI of the directory where artifacts should be uploaded This can be a local path (starting with “/”),
#'   or a distributed file system (DFS) path, like s3://bucket/directory or dbfs:/my/directory. If not set, the local ./mlruns
#'   directory will be chosen by default.
#' @param entry_point_name Name of the entry point for the run.
#' @param run_tags Additional metadata for run in key-value pairs.
#' @export
mlflow_create_run <- function(experiment_id = NULL, user_id = NULL, run_name = NULL,
                              source_type = NULL, source_name = NULL, status = NULL,
                              start_time = NULL, end_time = NULL, source_version = NULL,
                              artifact_uri = NULL, entry_point_name = NULL, run_tags = NULL) {
  response <- mlflow_rest("runs", "create", verb = "POST", data = list(
    experiment_id = experiment_id,
    user_id = user_id,
    run_name = run_name,
    source_type = source_type,
    source_name = source_name,
    status = status,
    start_time = start_time,
    end_time = end_time,
    source_version = source_version,
    artifact_uri = artifact_uri,
    entry_point_name = entry_point_name,
    run_tags = run_tags
  ))
  as.data.frame(response$run$info)
}

mlflow_relative_paths <- function(paths) {
  gsub(paste0("^", file.path(getwd(), "")), "", paths)
}

#' Active Experiment
#'
#' Creates an MLflow experiment and makes it active.
#'
#' @param name The name of the experiment to create.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' # activates experiment
#' mlflow_experiment("My Experiment")
#'
#' # activates experiment in remote MLflow server
#' mlflow_tracking_url("http://tracking-server:5000")
#' mlflow_experiment("My Experiment")
#' }
#'
#' @export
mlflow_experiment <- function(name) {
  if (!name %in% mlflow_list_experiments()$name) {
    mlflow_create_experiment(name)
  }

  exps <- mlflow_list_experiments()
  experiment_id <- exps[exps$name == "Test",]$experiment_id

  Sys.setenv(MLFLOW_EXPERIMENT_ID = experiment_id)

  invisible(experiment_id)
}

#' Log to MLflow
#'
#' Logs a value to MLflow for the active run.
#'
#' @param name The name to identify this log entry.
#' @param value The value to log into this entry.
#'
#' @export
mlflow_log <- function(name, value) {
  invisible(NULL)
}
