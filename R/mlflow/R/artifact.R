mlflow_active_artifact_uri <- function() {
  run_info <- mlflow_get_run(mlflow_active_run())
  run_info$info$artifact_uri
}

mlflow_artifact_uri <- function(artifact_uri) {
  artifact_type <- NULL
  if (dir.exists(artifact_uri)) {
    artifact_type <- "local_artifact"
  }
  else {
    matches <- regexec("([^:]+):.*", artifact_uri)
    match <- regmatches(artifact_uri, matches)

    if (length(match[[1]]) < 2)
      stop("Artifact URI not recognized as a directory nor supported protocol.")

    protocol <- match[[1]][[2]]

    artifact_type <- switch(
      protocol,
      s3 = "s3_artifact",
      s3a = "s3_artifact",
      s3n = "s3_artifact",
      gs = "google_artifact",
      wasb = "azure_artifact",
      stop("Storing artifacts using '", protocol, "' is not supported.")
    )
  }

  artifact_type
}

#' Log Artifact
#'
#' Logs an specific file or directory as an artifact.
#'
#' @param path The file or directory to log as an artifact.
#'
#' @export
mlflow_log_artifact <- function(path) {

  artifact_uri <- mlflow_active_artifact_uri()
  artifact_type <- mlflow_artifact_type(artifact_uri)

  artifact <- structure(
    class = c(artifact_type),
    list(
      path = path,
      artifact_uri = artifact_uri
    )
  )

  mlflow_log_artifact_impl(artifact)
}

mlflow_log_artifact_impl <- function(artifact) {
  UseMethod("mlflow_save_artifact")
}

mlflow_log_artifact_impl.local_artifact <- function(artifact) {
  stop("Not implemented.")
}

mlflow_log_artifact_impl.s3_artifact <- function(artifact) {
  stop("Not implemented.")
}

mlflow_log_artifact_impl.google_artifact <- function(artifact) {
  stop("Not implemented.")
}

mlflow_log_artifact_impl.azure_artifact <- function(artifact) {
  stop("Not implemented.")
}
