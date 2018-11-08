#' Active Run
#'
#' Retrieves the active run.
#'
#' @export
mlflow_active_run <- function() {
  .globals$active_run
}

mlflow_set_active_run <- function(run) {
  .globals$active_run <- run
  invisible(run)
}

mlflow_get_active_experiment_id <- function() {
  .globals$active_experiment_id
}

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
#' Gets the remote tracking URI.
#'
#' @export
mlflow_get_tracking_uri <- function() {
  .globals$tracking_uri %||% {
    env_uri <- Sys.getenv("MLFLOW_TRACKING_URI")
    if (nchar(env_uri)) env_uri else fs::path_abs("mlruns")
  }
}
