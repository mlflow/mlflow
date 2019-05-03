#' Log Metric
#'
#' Logs a metric for a run. Metrics key-value pair that records a single float measure.
#'   During a single execution of a run, a particular metric can be logged several times.
#'   Backend will keep track of historical values along with timestamps.
#'
#' @param key Name of the metric.
#' @param value Float value for the metric being logged.
#' @param timestamp Unix timestamp in milliseconds at the time metric was logged.
#' @template roxlate-run-id
#' @template roxlate-client
#' @export
mlflow_log_metric <- function(key, value, timestamp = NULL, run_id = NULL, client = NULL) {
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)
  key <- cast_string(key)
  value <- cast_scalar_double(value)
  timestamp <- cast_nullable_scalar_double(timestamp)
  timestamp <- timestamp %||% current_time()
  mlflow_rest("runs", "log-metric", client = client, verb = "POST", data = list(
    run_uuid = run_id,
    run_id = run_id,
    key = key,
    value = value,
    timestamp = timestamp
  ))
  invisible(value)
}




mlflow_create_run <- function(user_id = NULL, run_name = NULL, source_type = NULL,
                              source_name = NULL, entry_point_name = NULL, start_time = NULL,
                              source_version = NULL, tags = NULL, experiment_id = NULL, client) {
  experiment_id <- resolve_experiment_id(experiment_id)
  tags <- if (!is.null(tags)) tags %>%
    purrr::imap(~ list(key = .y, value = .x)) %>%
    unname()

  start_time <- start_time %||% current_time()
  user_id <- user_id %||% mlflow_user()

  response <- mlflow_rest(
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

  mlflow_get_run(run_id = response$run$info$run_uuid, client = client)
}

#' Delete a Run
#'
#' @template roxlate-client
#' @template roxlate-run-id
#' @export
mlflow_delete_run <- function(run_id, client = NULL) {
  run_id <- cast_string(run_id)
  if (identical(run_id, mlflow_get_active_run_id()))
    stop("Cannot delete an active run.", call. = FALSE)
  client <- resolve_client(client)
  mlflow_rest("runs", "delete", client = client, verb = "POST", data = list(
    run_id = run_id
  ))
  invisible(NULL)
}

#' Restore a Run
#'
#' @template roxlate-client
#' @template roxlate-run-id
#' @export
mlflow_restore_run <- function(run_id, client = NULL) {
  run_id <- cast_string(run_id)
  client <- resolve_client(client)
  mlflow_rest("runs", "restore", client = client, verb = "POST", data = list(
    run_id = run_id
  ))
  mlflow_get_run(run_id)
}

#' Get Run
#'
#' Gets metadata, params, tags, and metrics for a run. In the case where multiple metrics with the
#' same key are logged for the run, returns only the value with the latest timestamp. If there are
#' multiple values with the latest timestamp, returns the maximum of these values.
#'
#' @template roxlate-run-id
#' @template roxlate-client
#' @export
mlflow_get_run <- function(run_id = NULL, client = NULL) {
  run_id <- resolve_run_id(run_id)
  client <- resolve_client(client)
  response <- mlflow_rest(
    "runs", "get",
    client = client, verb = "GET",
    query = list(run_uuid = run_id, run_id = run_id)
  )
  parse_run(response$run)
}

#' Log Batch
#'
#' Log a batch of metrics, params, and/or tags for a run. The server will respond with an error (non-200 status code)
#'   if any data failed to be persisted. In case of error (due to internal server error or an invalid request), partial
#'   data may be written.
#' @template roxlate-client
#' @template roxlate-run-id
#' @param metrics A named list of metrics to log.
#' @param params A named list of params to log.
#' @param tags A named list of tags to log.
#' @param timestamps (Optional) A list of timestamps of the same length as `metrics`.
#' @export
mlflow_log_batch <- function(metrics = NULL, params = NULL, tags = NULL, timestamps = NULL,
                             run_id = NULL, client = NULL) {
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)

  metrics <- construct_batch_list(metrics)
  params <- construct_batch_list(params)
  tags <- construct_batch_list(tags)

  if (!is.null(metrics)) {
    metrics <- if (is.null(timestamps)) {
      purrr::map(metrics, ~ c(.x, timestamp = current_time()))
    } else {
      if (length(metrics) != length(timestamps))
        stop("`metrics` and `timestamps` must be of the same length.", call. = FALSE)
      timestamps <- purrr::map(timestamps, ~ list(timestamp = .x))
      purrr::map2(metrics, timestamps, c)
    }
  }

  mlflow_rest("runs", "log-batch", client = client, verb = "POST", data = list(
    run_id = run_id,
    metrics = metrics,
    params = params,
    tags = tags
  ))

  invisible(NULL)
}

construct_batch_list <- function(l) {
  if (is.null(l)) {
    l
  } else {
    l %>%
      purrr::imap(~ list(key = .y, value = .x)) %>%
      unname()
  }
}

#' Set Tag
#'
#' Sets a tag on a run. Tags are run metadata that can be updated during a run and
#'  after a run completes.
#'
#' @param key Name of the tag. Maximum size is 255 bytes. This field is required.
#' @param value String value of the tag being logged. Maximum size is 500 bytes. This field is required.
#' @template roxlate-run-id
#' @template roxlate-client
#' @export
mlflow_set_tag <- function(key, value, run_id = NULL, client = NULL) {
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)

  key <- cast_string(key)
  value <- cast_string(value)

  mlflow_rest("runs", "set-tag", client = client, verb = "POST", data = list(
    run_uuid = run_id,
    run_id = run_id,
    key = key,
    value = value
  ))

  invisible(NULL)
}

#' Log Parameter
#'
#' Logs a parameter for a run. Examples are params and hyperparams
#'   used for ML training, or constant dates and values used in an ETL pipeline.
#'   A param is a STRING key-value pair. For a run, a single parameter is allowed
#'   to be logged only once.
#'
#' @param key Name of the parameter.
#' @param value String value of the parameter.
#' @template roxlate-run-id
#' @template roxlate-client
#' @export
mlflow_log_param <- function(key, value, run_id = NULL, client = NULL) {
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)

  key <- cast_string(key)
  value <- cast_string(value)

  mlflow_rest("runs", "log-parameter", client = client, verb = "POST", data = list(
    run_uuid = run_id,
    run_id = run_id,
    key = key,
    value = cast_string(value)
  ))

  invisible(value)
}

#' Get Metric History
#'
#' Get a list of all values for the specified metric for a given run.
#'
#' @template roxlate-run-id
#' @template roxlate-client
#' @param metric_key Name of the metric.
#'
#' @export
mlflow_get_metric_history <- function(metric_key, run_id = NULL, client = NULL) {
  run_id <- resolve_run_id(run_id)
  client <- resolve_client(client)

  metric_key <- cast_string(metric_key)

  response <- mlflow_rest(
    "metrics", "get-history",
    client = client, verb = "GET",
    query = list(run_uuid = run_id, run_id = run_id, metric_key = metric_key)
  )

  response$metrics %>%
    purrr::transpose() %>%
    purrr::map(unlist) %>%
    purrr::map_at("timestamp", milliseconds_to_date) %>%
    tibble::as_tibble()
}

#' Search Runs
#'
#' Search for runs that satisfy expressions. Search expressions can use Metric and Param keys.
#'
#' @template roxlate-client
#' @param experiment_ids List of experiment IDs to search over. Attempts to use active experiment if not specified.
#' @param filter A filter expression over params, metrics, and tags, allowing returning a subset of runs.
#'   The syntax is a subset of SQL which allows only ANDing together binary operations between a param/metric/tag and a constant.
#' @param run_view_type Run view type.
#'
#' @export
mlflow_search_runs <- function(filter = NULL,
                               run_view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL"), experiment_ids = NULL,
                               client = NULL) {
  experiment_ids <- resolve_experiment_id(experiment_ids)
  client <- resolve_client(client)

  run_view_type <- match.arg(run_view_type)
  experiment_ids <- cast_double_list(experiment_ids)
  filter <- cast_nullable_string(filter)

  response <- mlflow_rest("runs", "search", client = client, verb = "POST", data = list(
    experiment_ids = experiment_ids,
    filter = filter,
    run_view_type = run_view_type
  ))

  response$run %>%
    purrr::map_df(parse_run)
}

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

  response$files %>%
    purrr::transpose() %>%
    purrr::map(unlist) %>%
    tibble::as_tibble()
}

mlflow_set_terminated <- function(status, end_time, run_id, client) {

  response <- mlflow_rest("runs", "update", verb = "POST", client = client, data = list(
    run_uuid = run_id,
    run_id = run_id,
    status = status,
    end_time = end_time
  ))
  mlflow_get_run(client = client, run_id = response$run_info$run_uuid)
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



#' List Run Infos
#'
#' List run infos.
#'
#' @param experiment_id Experiment ID. Attempts to use the active experiment if not specified.
#' @param run_view_type Run view type.
#' @template roxlate-client
#' @export
mlflow_list_run_infos <- function(run_view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL"),
                                  experiment_id = NULL, client = NULL) {
  experiment_id <- resolve_experiment_id(experiment_id)
  client <- resolve_client(client)

  run_view_type <- match.arg(run_view_type)
  experiment_ids <- cast_string_list(experiment_id)

  response <- mlflow_rest("runs", "search", client = client, verb = "POST", data = list(
    experiment_ids = experiment_ids,
    filter = NULL,
    run_view_type = run_view_type
  ))

  response$runs %>%
    purrr::map("info") %>%
    purrr::map_df(parse_run_info)
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

  mlflow_list_artifacts(run_id = run_id, path = artifact_path, client = client)
}


#' Start Run
#'
#' Starts a new run. If `client` is not provided, this function infers contextual information such as
#'   source name and version, and also registers the created run as the active run. If `client` is provided,
#'   no inference is done, and additional arguments such as `user_id` and `start_time` can be provided.
#'
#' @param run_id If specified, get the run with the specified UUID and log metrics
#'   and params under that run. The run's end time is unset and its status is set to
#'   running, but the run's other attributes remain unchanged.
#' @param experiment_id Used only when `run_id` is unspecified. ID of the experiment under
#'   which to create the current run. If unspecified, the run is created under
#'   a new experiment with a randomly generated name.
#' @param source_name Name of the source file or URI of the project to be associated with the run.
#'   Defaults to the current file if none provided.
#' @param source_version Optional Git commit hash to associate with the run.
#' @param entry_point_name Optional name of the entry point for to the current run.
#' @param source_type Integer enum value describing the type of the run  ("local", "project", etc.).
#' @param user_id User ID or LDAP for the user executing the run. Only used when `client` is specified.
#' @param run_name Human readable name for run. Only used when `client` is specified.
#' @param start_time Unix timestamp of when the run started in milliseconds. Only used when `client` is specified.
#' @param tags Additional metadata for run in key-value pairs. Only used when `client` is specified.
#' @template roxlate-client
#'
#' @examples
#' \dontrun{
#' with(mlflow_start_run(), {
#'   mlflow_log("test", 10)
#' })
#' }
#'
#' @export
mlflow_start_run <- function(run_id = NULL, experiment_id = NULL, source_name = NULL,
                             source_version = NULL, entry_point_name = NULL,
                             source_type = NULL, user_id = NULL, run_name = NULL, start_time = NULL,
                             tags = NULL, client = NULL) {

  # When `client` is provided, this function acts as a wrapper for `runs/create` and does not register
  #  an active run.
  if (!is.null(client)) {
    if (!is.null(run_id)) stop("`run_id` should not be specified when `client` is specified.", call. = FALSE)
    run <- mlflow_create_run(client = client, user_id = user_id, run_name = run_name, source_type = source_type,
                             source_name = source_name, entry_point_name = entry_point_name, start_time = start_time,
                             source_version = source_version, tags = tags, experiment_id = experiment_id)
    return(run)
  }

  # Fluent mode, check to see if extraneous params passed.

  if (!is.null(user_id)) stop("`user_id` should only be specified when `client` is specified.", call. = FALSE)
  if (!is.null(run_name)) stop("`run_name` should only be specified when `client` is specified.", call. = FALSE)
  if (!is.null(start_time)) stop("`start_time` should only be specified when `client` is specified.", call. = FALSE)
  if (!is.null(tags)) stop("`tags` should only be specified when `client` is specified.", call. = FALSE)

  source_type <- source_type %||% "LOCAL"
  active_run_id <- mlflow_get_active_run_id()
  if (!is.null(active_run_id)) {
    stop("Run with UUID ", active_run_id, " is already active.",
         call. = FALSE
    )
  }

  existing_run_id <- run_id %||% {
    env_run_id <- Sys.getenv("MLFLOW_RUN_ID")
    if (nchar(env_run_id)) env_run_id
  }

  client <- mlflow_client()

  run <- if (!is.null(existing_run_id)) {
    # This is meant to pick up existing run when we're inside `mlflow_source()` called via `mlflow run`.
    mlflow_get_run(client = client, run_id = existing_run_id)
  } else {
    experiment_id <- mlflow_infer_experiment_id(experiment_id)
    client <- mlflow_client()

    args <- mlflow_get_run_context(
      client,
      experiment_id = experiment_id,
      source_name = source_name,
      source_version = source_version,
      entry_point_name = entry_point_name,
      source_type = source_type
    )
    do.call(mlflow_create_run, args)
  }
  mlflow_set_active_run_id(mlflow_id(run))
  run
}


mlflow_get_run_context <- function(client, ...) {
  UseMethod("mlflow_get_run_context")
}

mlflow_get_run_context.default <- function(client, source_name, source_version, experiment_id,
                                           ...) {
  list(
    client = client,
    source_name = source_name %||% get_source_name(),
    source_version = source_version %||% get_source_version(),
    experiment_id = experiment_id %||% 0,
    ...
  )
}

#' End a Run
#'
#' Terminates a run. Attempts to end the current active run if `run_id` is not specified.
#'
#' @param status Updated status of the run. Defaults to `FINISHED`.
#' @param end_time Unix timestamp of when the run ended in milliseconds.
#' @template roxlate-run-id
#' @template roxlate-client
#'
#' @export
mlflow_end_run <- function(status = c("FINISHED", "SCHEDULED", "FAILED", "KILLED"),
                           end_time = NULL, run_id = NULL, client = NULL) {

  status <- match.arg(status)
  end_time <- end_time %||% current_time()

  active_run_id <- mlflow_get_active_run_id()

  if (!is.null(client) && is.null(run_id))
    stop("`run_id` must be specified when `client` is specified.", call. = FALSE)

  run <- if (!is.null(run_id)) {
    client <- resolve_client(client)
    mlflow_set_terminated(client = client, run_id = run_id, status = status,
                          end_time = end_time)
  } else {
    if (is.null(active_run_id)) stop("There is no active run to end.", call. = FALSE)
    client <- mlflow_client()
    run_id <- active_run_id
    mlflow_set_terminated(client = client, run_id = active_run_id, status = status,
                          end_time = end_time)
  }

  if (identical(run_id, active_run_id)) mlflow_set_active_run_id(NULL)
  run
}
