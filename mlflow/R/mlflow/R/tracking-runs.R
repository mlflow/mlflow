#' @include tracking-globals.R
NULL

# Translate metric to value to safe format for REST.
metric_value_to_rest <- function(value) {
  if (is.nan(value)) {
    as.character(NaN)
  } else if (value == Inf) {
    "Infinity"
  } else if (value == -Inf) {
    "-Infinity"
  } else {
    as.character(value)
  }
}

#' Log Metric
#'
#' Logs a metric for a run. Metrics key-value pair that records a single float measure.
#'   During a single execution of a run, a particular metric can be logged several times.
#'   The MLflow Backend keeps track of historical metric values along two axes: timestamp and step.
#'
#' @param key Name of the metric.
#' @param value Float value for the metric being logged.
#' @param timestamp Timestamp at which to log the metric. Timestamp is rounded to the nearest
#'  integer. If unspecified, the number of milliseconds since the Unix epoch is used.
#' @param step Step at which to log the metric. Step is rounded to the nearest integer. If
#'  unspecified, the default value of zero is used.
#' @template roxlate-run-id
#' @template roxlate-client
#' @export
mlflow_log_metric <- function(key, value, timestamp = NULL, step = NULL, run_id = NULL,
                              client = NULL) {
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)
  key <- cast_string(key)
  value <- cast_scalar_double(value, allow_na = TRUE)
  # convert Inf to 'Infinity'
  value <- metric_value_to_rest(value)
  timestamp <- cast_nullable_scalar_double(timestamp)
  timestamp <- round(timestamp %||% current_time())
  step <- round(cast_nullable_scalar_double(step) %||% 0)
  data <- list(
    run_uuid = run_id,
    run_id = run_id,
    key = key,
    value = value,
    timestamp = timestamp,
    step = step
  )
  mlflow_rest("runs", "log-metric", client = client, verb = "POST", data = data)
  mlflow_register_tracking_event("log_metric", data)

  invisible(value)
}

mlflow_create_run <- function(start_time = NULL, tags = NULL, experiment_id = NULL, client) {
  experiment_id <- resolve_experiment_id(experiment_id)

  # Read user_id from tags
  # user_id is deprecated and will be removed from a future release
  user_id <- tags[[MLFLOW_TAGS$MLFLOW_USER]] %||% "unknown"

  tags <- if (!is.null(tags)) tags %>%
    purrr::imap(~ list(key = .y, value = .x)) %>%
    unname()

  start_time <- start_time %||% current_time()

  data <- list(
    experiment_id = experiment_id,
    user_id = user_id,
    start_time = start_time,
    tags = tags
  )
  response <- mlflow_rest(
    "runs", "create", client = client, verb = "POST", data = data
  )
  run_id <- response$run$info$run_uuid
  data$run_id <- run_id
  mlflow_register_tracking_event("create_run", data)

  mlflow_get_run(run_id = run_id, client = client)
}

#' Delete a Run
#'
#' Deletes the run with the specified ID.
#' @template roxlate-client
#' @template roxlate-run-id
#' @export
mlflow_delete_run <- function(run_id, client = NULL) {
  run_id <- cast_string(run_id)
  if (identical(run_id, mlflow_get_active_run_id()))
    stop("Cannot delete an active run.", call. = FALSE)
  client <- resolve_client(client)
  data <- list(run_id = run_id)
  mlflow_rest("runs", "delete", client = client, verb = "POST", data = data)
  mlflow_register_tracking_event("delete_run", data)
  invisible(NULL)
}

#' Restore a Run
#'
#' Restores the run with the specified ID.
#' @template roxlate-client
#' @template roxlate-run-id
#' @export
mlflow_restore_run <- function(run_id, client = NULL) {
  run_id <- cast_string(run_id)
  client <- resolve_client(client)
  data <- list(run_id = run_id)
  mlflow_rest("runs", "restore", client = client, verb = "POST", data = data)
  mlflow_register_tracking_event("restore_run", data)

  mlflow_get_run(run_id)
}

#' Get Run
#'
#' Gets metadata, params, tags, and metrics for a run. Returns a single value for each metric
#' key: the most recently logged metric value at the largest step.
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
#' @param metrics A dataframe of metrics to log, containing the following columns: "key", "value",
#'  "step", "timestamp". This dataframe cannot contain any missing ('NA') entries.
#' @param params A dataframe of params to log, containing the following columns: "key", "value".
#'  This dataframe cannot contain any missing ('NA') entries.
#' @param tags A dataframe of tags to log, containing the following columns: "key", "value".
#'  This dataframe cannot contain any missing ('NA') entries.
#' @export
mlflow_log_batch <- function(metrics = NULL, params = NULL, tags = NULL, run_id = NULL,
                             client = NULL) {
  validate_batch_input("metrics", metrics, c("key", "value", "step", "timestamp"))
  metrics$value <- unlist(lapply(metrics$value, metric_value_to_rest))
  validate_batch_input("params", params, c("key", "value"))
  validate_batch_input("tags", tags, c("key", "value"))

  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)

  data <- list(
    run_id = run_id,
    metrics = metrics,
    params = params,
    tags = tags
  )
  mlflow_rest("runs", "log-batch", client = client, verb = "POST", data = data)
  mlflow_register_tracking_event("log_batch", data)

  invisible(NULL)
}

has_nas <- function(df) {
  any(is.na(df[, which(names(df) != "value")])) ||
  any(is.na(df$value) & !is.nan(df$value))
}

validate_batch_input <- function(input_type, input_dataframe, expected_column_names) {
  if (is.null(input_dataframe)) {
    return()
  } else if (!setequal(names(input_dataframe), expected_column_names)) {
    msg <- paste(input_type,
                 " batch input dataframe must contain exactly the following columns: ",
                 paste(expected_column_names, collapse = ", "),
                 ". Found: ",
                 paste(names(input_dataframe), collapse = ", "),
                 sep = "")
    stop(msg, call. = FALSE)
  } else if (has_nas(input_dataframe)) {
    msg <- paste(input_type,
                 " batch input dataframe contains a missing ('NA') entry.",
                 sep = "")
    stop(msg, call. = FALSE)
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

  data <- list(
    run_uuid = run_id,
    run_id = run_id,
    key = key,
    value = value
  )
  mlflow_rest("runs", "set-tag", client = client, verb = "POST", data = data)
  mlflow_register_tracking_event("set_tag", data)

  invisible(NULL)
}

#' Delete Tag
#'
#' Deletes a tag on a run. This is irreversible. Tags are run metadata that can be updated during a run and
#'  after a run completes.
#'
#' @param key Name of the tag. Maximum size is 255 bytes. This field is required.
#' @template roxlate-run-id
#' @template roxlate-client
#' @export
mlflow_delete_tag <- function(key, run_id = NULL, client = NULL) {
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)

  key <- cast_string(key)

  data <- list(run_id = run_id, key = key)
  mlflow_rest("runs", "delete-tag", client = client, verb = "POST", data = data)
  mlflow_register_tracking_event("delete_tag", data)

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

  data <- list(
    run_uuid = run_id,
    run_id = run_id,
    key = key,
    value = cast_string(value)
  )
  mlflow_rest("runs", "log-parameter", client = client, verb = "POST", data = data)
  mlflow_register_tracking_event("log_param", data)

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
    purrr::map_at("step", as.double) %>%
    tibble::as_tibble()
}

#' Search Runs
#'
#' Search for runs that satisfy expressions. Search expressions can use Metric and Param keys.
#'
#' @template roxlate-client
#' @param experiment_ids List of string experiment IDs (or a single string experiment ID) to search
#' over. Attempts to use active experiment if not specified.
#' @param filter A filter expression over params, metrics, and tags, allowing returning a subset of runs.
#'   The syntax is a subset of SQL which allows only ANDing together binary operations between a param/metric/tag and a constant.
#' @param run_view_type Run view type.
#' @param order_by List of properties to order by. Example: "metrics.acc DESC".
#'
#' @export
mlflow_search_runs <- function(filter = NULL,
                               run_view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL"),
                               experiment_ids = NULL,
                               order_by = list(),
                               client = NULL) {
  experiment_ids <- resolve_experiment_id(experiment_ids)
  # If we get back a single experiment ID, e.g. the active experiment ID, convert it to a list
  if (is.atomic(experiment_ids)) {
    experiment_ids <- list(experiment_ids)
  }
  client <- resolve_client(client)

  run_view_type <- match.arg(run_view_type)
  experiment_ids <- cast_string_list(experiment_ids)
  filter <- cast_nullable_string(filter)

  response <- mlflow_rest("runs", "search", client = client, verb = "POST", data = list(
    experiment_ids = experiment_ids,
    filter = filter,
    run_view_type = run_view_type,
    order_by = cast_string_list(order_by)
  ))

  runs_list <- response$run %>%
    purrr::map(parse_run)
  do.call("rbind", runs_list) %||% data.frame()
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

mlflow_set_terminated <- function(status, end_time, run_id, client) {
  data <- list(
    run_uuid = run_id,
    run_id = run_id,
    status = status,
    end_time = end_time
  )
  response <- mlflow_rest("runs", "update", verb = "POST", client = client, data = data)
  mlflow_register_tracking_event("set_terminated", data)

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

# ' Download Artifacts from URI.
mlflow_download_artifacts_from_uri <- function(artifact_uri, client = mlflow_client()) {
  result <- mlflow_cli("artifacts", "download", "-u", artifact_uri, echo = FALSE, client = client)
  gsub("\n", "", result$stdout)
}

#' List Run Infos
#'
#' Returns a tibble whose columns contain run metadata (run ID, etc) for all runs under the
#' specified experiment.
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

  run_infos_list <- response$runs %>%
    purrr::map("info") %>%
    purrr::map(parse_run_info)
  do.call("rbind", run_infos_list) %||% data.frame()
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

# Record logged model metadata with the tracking server.
mlflow_record_logged_model <- function(model_spec, run_id = NULL, client = NULL) {
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)
  mlflow_rest("runs", "log-model", client = client, verb = "POST", data = list(
    run_id = run_id,
    model_json = jsonlite::toJSON(model_spec, auto_unbox=TRUE)
  ))
}

#' Start Run
#'
#' Starts a new run. If `client` is not provided, this function infers contextual information such as
#'   source name and version, and also registers the created run as the active run. If `client` is provided,
#'   no inference is done, and additional arguments such as `start_time` can be provided.
#'
#' @param run_id If specified, get the run with the specified UUID and log metrics
#'   and params under that run. The run's end time is unset and its status is set to
#'   running, but the run's other attributes remain unchanged.
#' @param experiment_id Used only when `run_id` is unspecified. ID of the experiment under
#'   which to create the current run. If unspecified, the run is created under
#'   a new experiment with a randomly generated name.
#' @param start_time Unix timestamp of when the run started in milliseconds. Only used when `client` is specified.
#' @param tags Additional metadata for run in key-value pairs. Only used when `client` is specified.
#' @param nested Controls whether the run to be started is nested in a parent run. `TRUE` creates a nest run.
#' @template roxlate-client
#'
#' @examples
#' \dontrun{
#' with(mlflow_start_run(), {
#'   mlflow_log_metric("test", 10)
#' })
#' }
#'
#' @export
mlflow_start_run <- function(run_id = NULL, experiment_id = NULL, start_time = NULL, tags = NULL, client = NULL, nested = FALSE) {

  # When `client` is provided, this function acts as a wrapper for `runs/create` and does not register
  #  an active run.
  if (!is.null(client)) {
    if (!is.null(run_id)) stop("`run_id` should not be specified when `client` is specified.", call. = FALSE)
    run <- mlflow_create_run(client = client, start_time = start_time,
                             tags = tags, experiment_id = experiment_id)
    return(run)
  }

  # Fluent mode, check to see if extraneous params passed.

  if (!is.null(start_time)) stop("`start_time` should only be specified when `client` is specified.", call. = FALSE)
  if (!is.null(tags)) stop("`tags` should only be specified when `client` is specified.", call. = FALSE)

  active_run_id <- mlflow_get_active_run_id()
  if (!is.null(active_run_id) && !nested) {
    stop("Run with UUID ",
         active_run_id,
         " is already active. To start a nested run, Call `mlflow_start_run()` with `nested = TRUE`.",
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
      experiment_id = experiment_id
    )
    do.call(mlflow_create_run, args)
  }
  mlflow_push_active_run_id(mlflow_id(run))
  mlflow_set_experiment(experiment_id = run$experiment_id)
  run
}

mlflow_get_run_context <- function(client, ...) {
  UseMethod("mlflow_get_run_context")
}

mlflow_get_run_context.default <- function(client, experiment_id, ...) {
  tags <- list()
  tags[[MLFLOW_TAGS$MLFLOW_USER]] <- mlflow_user()
  tags[[MLFLOW_TAGS$MLFLOW_SOURCE_NAME]] <- get_source_name()
  tags[[MLFLOW_TAGS$MLFLOW_SOURCE_VERSION]] <- get_source_version()
  tags[[MLFLOW_TAGS$MLFLOW_SOURCE_TYPE]] <- MLFLOW_SOURCE_TYPE$LOCAL
  list(
    client = client,
    tags = tags,
    experiment_id = experiment_id %||% 0,
    ...
  )
}

#' End a Run
#'
#' Terminates a run. Attempts to end the current active run if `run_id` is not specified.
#'
#' @param status Updated status of the run. Defaults to `FINISHED`. Can also be set to
#' "FAILED" or "KILLED".
#' @param end_time Unix timestamp of when the run ended in milliseconds.
#' @template roxlate-run-id
#' @template roxlate-client
#'
#' @export
mlflow_end_run <- function(status = c("FINISHED", "FAILED", "KILLED"),
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

  if (identical(run_id, active_run_id)) mlflow_pop_active_run_id()
  run
}

MLFLOW_TAGS <- list(
  MLFLOW_USER = "mlflow.user",
  MLFLOW_SOURCE_NAME = "mlflow.source.name",
  MLFLOW_SOURCE_VERSION = "mlflow.source.version",
  MLFLOW_SOURCE_TYPE = "mlflow.source.type"
)
