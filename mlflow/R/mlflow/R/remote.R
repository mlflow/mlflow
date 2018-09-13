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
  .globals$active_connection <- NULL
  .globals$active_experiment <- NULL
  .globals$active_run <- NULL

  invisible(uri)
}

#' Get Remote Tracking URI
#'
#' @export
mlflow_tracking_uri <- function() {
  .globals$tracking_uri %||% {
    env_uri <- Sys.getenv("MLFLOW_TRACKING_URI")
    if (nchar(env_uri)) env_uri else fs::path_abs("mlruns")
  }
}
