#' Active Run
#'
#' Retrieves or sets the active run.
#'
#' @name active_run
#' @export
mlflow_get_active_run <- function() {
  .globals$active_run
}

#' @rdname active_run
#' @param run The run object to make active.
#' @export
mlflow_set_active_run <- function(run) {
  .globals$active_run <- run
  invisible(run)
}

#' Active Experiment
#'
#' Retrieve or set the active experiment.
#'
#' @name active_experiment
#' @export
mlflow_get_active_experiment_id <- function() {
  .globals$active_experiment_id
}

#' @rdname active_experiment
#' @param experiment_id Identifer to get an experiment.
#' @export
mlflow_set_active_experiment_id <- function(experiment_id) {
  if (!identical(experiment_id, .globals$active_experiment_id)) {
    .globals$active_experiment_id <- experiment_id
  }

  invisible(experiment_id)
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
#' @export
mlflow_get_tracking_uri <- function() {
  .globals$tracking_uri %||% {
    env_uri <- Sys.getenv("MLFLOW_TRACKING_URI")
    if (nchar(env_uri)) env_uri else fs::path_abs("mlruns")
  }
}
