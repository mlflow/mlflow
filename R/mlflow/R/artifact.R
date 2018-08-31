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
mlflow_log_artifact <- function(path, artifact_path = NULL, run_uuid = NULL) {
  run_uuid <- mlflow_ensure_run_id(run_uuid)
  artifact_uri <- mlflow_get_run(run_uuid)$info$artifact_uri
  artifact_type <- mlflow_artifact_type(artifact_uri)

  artifact_uri <- ifelse(
    is.null(artifact_path),
    artifact_uri,
    fs::path(artifact_uri, artifact_path)
  )

  artifact_uri <- structure(
    class = c(artifact_type),
    artifact_uri
  )

  if (dir.exists(path)) {
    for (file in dir(path, recursive = TRUE))
      mlflow_log_artifact_impl(artifact_uri, fs::path(path, file))
  }
  else {
    mlflow_log_artifact_impl(artifact_uri, path)
  }
}

mlflow_log_artifact_impl <- function(artifact_uri, path) {
  UseMethod("mlflow_log_artifact_impl")
}

mlflow_log_artifact_impl.local_artifact <- function(artifact_uri, path) {
  if (!dir.exists(artifact_uri)) {
    dir.create(artifact_uri, recursive = TRUE)
  }

  file.copy(path, artifact_uri, overwrite = TRUE)
  invisible(NULL)
}

mlflow_log_artifact_impl.s3_artifact <- function(artifact_uri, path) {
  file_name <- basename(path)

  bucket <- mlflow_parse_bucket(artifact_uri)

  aws.s3::put_object(
    path,
    object = fs::path(bucket$path, file_name),
    bucket = bucket$name,
    check_region = TRUE
  )

  invisible(NULL)
}

#' @importFrom utils installed.packages
mlflow_log_artifact_impl.google_artifact <- function(artifact_uri, path) {
  if (!"googleCloudStorageR" %in% installed.packages()) {
    stop("The package 'googleCloudStorageR' is currently required but not installed, ",
         "please install using install.packages(\"googleCloudStorageR\").")
  }

  gcs_upload <- get("gcs_upload", envir = asNamespace("googleCloudStorageR"))

  file_name <- basename(path)

  bucket <- mlflow_parse_bucket(artifact_uri)

  gcs_upload(file = path,
             bucket = bucket$name,
             name = fs::path(bucket$path, file_name))

  invisible(NULL)
}

mlflow_log_artifact_impl.azure_artifact <- function(artifact_uri, path) {
  bucket <- mlflow_parse_bucket(artifact_uri)

  processx::run("az",
                "storage",
                "blob",
                "upload",
                "--container-name",
                bucket$name,
                "--file",
                path,
                "--name",
                bucket$path,
                echo = TRUE)

  invisible(NULL)
}

mlflow_parse_bucket <- function(artifact_uri) {
  match <- regexec("[^:]+://([^/]+)(.*)", artifact_uri)
  results <- regmatches(artifact_uri, match)

  list(
    name = results[[1]][2],
    path = results[[1]][3]
  )
}
