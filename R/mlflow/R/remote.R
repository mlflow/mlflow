#' Remote Tracking URL
#'
#' Specifies the URL to the remote MLflow server that will be used
#' to track experiments.
#'
#' @param url The URL to the remote MLflow server.
#'
#' @export
mlflow_tracking_url <- function(url) {
  Sys.setenv(MLFLOW_TRACKING_URI = url)
}

mlflow_tracking_url_get <- function() {
  env_url <- Sys.getenv("MLFLOW_TRACKING_URI")

  if (startsWith(env_url, "http")) {
    env_url
  } else {
    mlflow_connection_url(mlflow_get_or_create_active_connection())
  }
}

mlflow_tracking_url_remote <- function() {
  Sys.getenv("MLFLOW_TRACKING_URI")
}

mlflow_tracking_is_remote <- function() {
  !identical(mlflow_tracking_url_remote(), "")
}
