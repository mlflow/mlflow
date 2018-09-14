#' Log Artifact
#'
#' Logs an specific file or directory as an artifact.
#'
#' @param path The file or directory to log as an artifact.
#' @param artifact_path Destination path within the run’s artifact URI.
#' @param run_uuid The run associated with this artifact.
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
  run_uuid <- run_uuid %||%
    mlflow_active_run()$run_info$run_uuid %||%
    stop("`run_uuid` must be specified when there is no active run.")

  artifact_param <- NULL
  if (!is.null(artifact_path)) artifact_param <- "--artifact-path"

  mlflow_cli("artifacts",
             "log-artifact",
             "--local-file",
             path,
             artifact_param,
             artifact_path,
             "--run-id",
             run_uuid)

  invisible(NULL)
}
