mlflow_get_run_id <- function(run) cast_nullable_string(run$run_uuid)

mlflow_set_active_run_id <- function(run_id) {
  .globals$active_run_id <- run_id
}

mlflow_get_active_run_id <- function() {
  .globals$active_run_id
}

mlflow_get_active_experiment_id <- function() {
  .globals$active_experiment$experiment_id
}

mlflow_get_active_experiment <- function() {
  .globals$active_experiment
}

mlflow_set_active_experiment <- function(experiment) {
  UseMethod("mlflow_set_active_experiment")
}

mlflow_set_active_experiment.character <- function(experiment) {
  experiment <- mlflow_get_experiment(experiment)
  .globals$active_experiment <- experiment
}

mlflow_set_active_experiment.mlflow_experiment <- function(experiment) {
  .globals$active_experiment <- experiment
}

mlflow_set_active_experiment.NULL <- function(experiment) {
  .globals$active_experiment <- NULL
}

#' Set Remote Tracking URI
#'
#' Specifies the URI to the remote MLflow server that will be used
#' to track experiments.
#'
#' @param uri The URI to the remote MLflow server.
#'
#' @export
mlflow_set_tracking_uri <- function(uri) {
  .globals$tracking_uri <- uri
  invisible(uri)
}

#' Get Remote Tracking URI
#'
#' Gets the remote tracking URI.
#'
#' @export
mlflow_get_tracking_uri <- function() {
  .globals$tracking_uri %||% {
    env_uri <- Sys.getenv("MLFLOW_TRACKING_URI")
    if (nchar(env_uri)) env_uri else paste("file://", fs::path_abs("mlruns"), sep = "")
  }
}
