#' List Artifacts
#'
#' Gets a list of artifacts.
#'
#' @template roxlate-client
#' @template roxlate-run-id
#' @param path The run's relative artifact path to list from. If not specified, it is
#'  set to the root artifact path
#'
#' @export
mlflow_list_artifacts <- function(path = NULL, run_id = NULL, client = NULL) {
  run_id <- resolve_run_id(run_id)
  client <- resolve_client(client)

  response <-   mlflow_rest(
    "artifacts", "list",
    client = client, verb = "GET",
    query = list(
      run_uuid = run_id,
      run_id = run_id,
      path = path
    )
  )

  message(glue::glue("Root URI: {uri}", uri = response$root_uri))

  files_list <- if (!is.null(response$files)) response$files else list()
  files_list <- purrr::map(files_list, function(file_info) {
    if (is.null(file_info$file_size)) {
      file_info$file_size <- NA
    }
    file_info
  })
  files_list %>%
    purrr::transpose() %>%
    purrr::map(unlist) %>%
    tibble::as_tibble()
}

#' Download Artifacts
#'
#' Download an artifact file or directory from a run to a local directory if applicable,
#'   and return a local path for it.
#'
#' @template roxlate-client
#' @template roxlate-run-id
#' @param path Relative source path to the desired artifact.
#' @export
mlflow_download_artifacts <- function(path, run_id = NULL, client = NULL) {
  run_id <- resolve_run_id(run_id)
  client <- resolve_client(client)
  result <- mlflow_cli(
    "artifacts", "download",
    "--run-id", run_id,
    "--artifact-path", path,
    echo = FALSE,
    stderr_callback = function(x, p) {
      if (grepl("FileNotFoundError", x)) {
        stop(
          gsub("(.|\n)*(?=FileNotFoundError)", "", x, perl = TRUE),
          call. = FALSE
        )
      }
    },
    client = client
  )
  gsub("\n", "", result$stdout)
}

# ' Download Artifacts from URI.
mlflow_download_artifacts_from_uri <- function(artifact_uri, client = mlflow_client()) {
  result <- mlflow_cli("artifacts", "download", "-u", artifact_uri, echo = FALSE, client = client)
  gsub("\n", "", result$stdout)
}

#' Log Artifact
#'
#' Logs a specific file or directory as an artifact for a run.
#'
#' @param path The file or directory to log as an artifact.
#' @param artifact_path Destination path within the run's artifact URI.
#' @template roxlate-client
#' @template roxlate-run-id
#'
#' @details
#'
#' When logging to Amazon S3, ensure that you have the s3:PutObject, s3:GetObject,
#' s3:ListBucket, and s3:GetBucketLocation permissions on your bucket.
#'
#' Additionally, at least the \code{AWS_ACCESS_KEY_ID} and \code{AWS_SECRET_ACCESS_KEY}
#' environment variables must be set to the corresponding key and secrets provided
#' by Amazon IAM.
#'
#' @export
mlflow_log_artifact <- function(path, artifact_path = NULL, run_id = NULL, client = NULL) {
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)
  artifact_param <- NULL
  if (!is.null(artifact_path)) artifact_param <- "--artifact-path"

  if (as.logical(fs::is_file(path))) {
    command <- "log-artifact"
    local_param <- "--local-file"
  } else {
    command <- "log-artifacts"
    local_param <- "--local-dir"
  }

  mlflow_cli("artifacts",
             command,
             local_param,
             path,
             artifact_param,
             artifact_path,
             "--run-id",
             run_id,
             client = client
  )

  invisible(mlflow_list_artifacts(run_id = run_id, path = artifact_path, client = client))
}