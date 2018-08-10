mlflow_active_artifact_uri <- function() {
  run_info <- mlflow_active_run()
  run_info$run_info$artifact_uri
}

mlflow_artifact_type <- function(artifact_uri) {
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
#' @param artifact_path Destination path within the run’s artifact URI.
#'
#' @details
#'
#' When logging to Amazon S3, ensure that the user has a proper policy
#' attach to it, for instance:
#'
#' \code{
#' {
#' "Version": "2012-10-17",
#' "Statement": [
#'   {
#'     "Sid": "VisualEditor0",
#'     "Effect": "Allow",
#'     "Action": [
#'       "s3:PutObject",
#'       "s3:GetObject",
#'       "s3:ListBucket",
#'       "s3:GetBucketLocation"
#'       ],
#'     "Resource": [
#'       "arn:aws:s3:::mlflow-test/*",
#'       "arn:aws:s3:::mlflow-test"
#'       ]
#'   }
#'   ]
#' }
#' }
#'
#' Additionally, at least the \code{AWS_ACCESS_KEY_ID} and \code{AWS_SECRET_ACCESS_KEY}
#' environment variables must be set to the corresponding key and secrets provided
#' by Amazon IAM.
#'
#' @export
mlflow_log_artifact <- function(path, artifact_path = NULL) {

  artifact_uri <- mlflow_active_artifact_uri()
  artifact_type <- mlflow_artifact_type(artifact_path)

  artifact_uri <- structure(
    class = c(artifact_type),
    artifact_path
  )

  if (dir.exists(path)) {
    for (file in dir(path, recursive = TRUE))
      mlflow_log_artifact_impl(artifact_uri, fs::path(path, file), artifact_path)
  }
  else {
    mlflow_log_artifact_impl(artifact_uri, path, artifact_path)
  }
}

mlflow_log_artifact_impl <- function(artifact_uri, path, artifact_path) {
  UseMethod("mlflow_log_artifact_impl")
}

mlflow_log_artifact_impl.local_artifact <- function(artifact_uri, path, artifact_path) {
  destination_path <- ifelse(
    is.null(artifact_path),
    artifact_uri,
    file.path(artifact_uri, artifact_path)
  )

  if (!dir.exists(destination_path)) {
    dir.create(destination_path, recursive = TRUE)
  }

  file.copy(path, destination_path)
  invisible(NULL)
}

mlflow_log_artifact_impl.s3_artifact <- function(artifact_uri, path, artifact_path) {
  mlflow_store_s3(path, artifact_uri)
}

mlflow_log_artifact_impl.google_artifact <- function(artifact_uri, path, artifact_path) {
  stop("Not implemented.")
}

mlflow_log_artifact_impl.azure_artifact <- function(artifact_uri, path, artifact_path) {
  stop("Not implemented.")
}
