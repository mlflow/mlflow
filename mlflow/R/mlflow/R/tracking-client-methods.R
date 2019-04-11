mlflow_client_create_experiment <- function(client, name, artifact_location) {
  mlflow_rest(
    "experiments", "create",
    client = client, verb = "POST",
    data = list(
      name = name,
      artifact_location = artifact_location
    )
  )
}

mlflow_client_list_experiments <- function(client, view_type) {
  mlflow_rest(
    "experiments", "list",
    client = client, verb = "GET",
    query = list(view_type = view_type)
  )
}

mlflow_client_get_experiment <- function(client, experiment_id) {
  mlflow_rest(
    "experiments", "get",
    client = client, query = list(experiment_id = experiment_id)
  )
}

#' Get Experiment by Name
#'
#' Get meta data for experiment by name.
#'
#' @param name The experiment name.
#' @template roxlate-client
mlflow_client_get_experiment_by_name <- function(client, name) {
  exps <- mlflow_client_list_experiments(client = client)
  if ("name" %in% names(exps) && length(exps$name)) {
    experiment <- exps[exps$name == name, ]
    if (nrow(experiment)) experiment else NULL
  } else {
    NULL
  }
}


mlflow_client_create_run <- function(client, experiment_id, user_id = NULL, run_name = NULL, source_type = NULL,
                                     source_name = NULL, entry_point_name = NULL, start_time = NULL,
                                     source_version = NULL, tags = NULL) {
  mlflow_rest(
    "runs", "create",
    client = client, verb = "POST",
    data = list(
      experiment_id = experiment_id,
      user_id = user_id,
      run_name = run_name,
      source_type = source_type,
      source_name = source_name,
      entry_point_name = entry_point_name,
      start_time = start_time,
      source_version = source_version,
      tags = tags
    )
  )
}

mlflow_rest_update_run <- function(client, run_uuid, status, end_time) {
  mlflow_rest("runs", "update", verb = "POST", client = client, data = list(
    run_uuid = run_uuid,
    status = status,
    end_time = end_time
  ))
}

mlflow_client_delete_experiment <- function(client, experiment_id) {
  mlflow_rest(
    "experiments", "delete",
    verb = "POST", client = client,
    data = list(experiment_id = experiment_id)
  )
}

mlflow_client_restore_experiment <- function(client, experiment_id) {
  mlflow_rest(
    "experiments", "restore",
    client = client, verb = "POST",
    data = list(experiment_id = experiment_id)
  )
}

mlflow_client_rename_experiment <- function(client, experiment_id, new_name) {
  mlflow_rest(
    "experiments", "update",
    client = client, verb = "POST",
    data = list(
      experiment_id = experiment_id,
      new_name = new_name
    )
  )
}

mlflow_client_get_run <- function(client, run_id) {
  response <- mlflow_rest(
    "runs", "get",
    client = client, verb = "GET",
    query = list(run_uuid = run_id)
  )
}

mlflow_client_log_metric <- function(client, run_id, key, value, timestamp) {
  mlflow_rest("runs", "log-metric", client = client, verb = "POST", data = list(
    run_uuid = run_id,
    key = key,
    value = value,
    timestamp = timestamp
  ))
}

mlflow_client_log_batch <- function(client, run_id, metrics, params, tags) {
  mlflow:::mlflow_rest("runs", "log-batch", client = client, verb = "POST", data = list(
    run_id = run_id,
    metrics = metrics,
    params = params,
    tags = tags
  ))
}

mlflow_client_log_param <- function(client, run_id, key, value) {
  mlflow_rest("runs", "log-parameter", client = client, verb = "POST", data = list(
    run_uuid = run_id,
    key = key,
    value = cast_string(value)
  ))
}

mlflow_client_set_tag <- function(client, run_id, key, value) {
  mlflow_rest("runs", "set-tag", client = client, verb = "POST", data = list(
    run_uuid = run_id,
    key = key,
    value = value
  ))
}

mlflow_client_get_metric_history <- function(client, run_id, metric_key) {
  response <- mlflow_rest(
    "metrics", "get-history",
    client = client, verb = "GET",
    query = list(run_uuid = run_id, metric_key = metric_key)
  )
}

mlflow_client_search_runs <- function(client, experiment_ids, filter, run_view_type) {
  mlflow_rest("runs", "search", client = client, verb = "POST", data = list(
    experiment_ids = experiment_ids,
    filter = filter,
    run_view_type = run_view_type
  ))
}

mlflow_client_delete_run <- function(client, run_id) {
  mlflow_rest("runs", "delete", client = client, verb = "POST", data = list(
    run_id = run_id
  ))
}

mlflow_client_restore_run <- function(client, run_id) {
  mlflow_rest("runs", "restore", client = client, verb = "POST", data = list(
    run_id = run_id
  ))
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
#' When logging to Amazon S3, ensure that the user has a proper policy
#' attached to it, for instance:
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
mlflow_client_log_artifact <- function(client, run_id, path, artifact_path = NULL) {
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

  invisible(NULL)
}

mlflow_client_list_artifacts <- function(client, run_id, path = NULL) {
  mlflow_rest(
    "artifacts", "list",
    client = client, verb = "GET",
    query = list(
      run_uuid = run_id,
      path = path
    )
  )
}

