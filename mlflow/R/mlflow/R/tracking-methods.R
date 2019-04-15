#' Create Experiment
#'
#' Creates an MLflow experiment.
#'
#' @param name The name of the experiment to create.
#' @param artifact_location Location where all artifacts for this experiment are stored. If
#'   not provided, the remote server will select an appropriate default.
#' @template roxlate-client
#' @export
mlflow_create_experiment <- function(name, artifact_location = NULL, client = NULL) {
  client <- client %||% mlflow_client()
  name <- forge::cast_string(name)
  response <- mlflow_rest(
    "experiments", "create",
    client = client, verb = "POST",
    data = list(
      name = name,
      artifact_location = artifact_location
    )
  )
  mlflow_get_experiment(client = client, experiment_id = response$experiment_id)
}

#' List Experiments
#'
#' Gets a list of all experiments.
#'
#' @param view_type Qualifier for type of experiments to be returned. Defaults to `ACTIVE_ONLY`.
#' @template roxlate-client
#' @export
mlflow_list_experiments <- function(view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL"), client = NULL) {
  client <- client %||% mlflow_client()
  view_type <- match.arg(view_type)
  response <-   mlflow_rest(
    "experiments", "list",
    client = client, verb = "GET",
    query = list(view_type = view_type)
  )

  # Return `NULL` if no experiments
  if (!length(response)) return(NULL)

  response$experiments %>%
    purrr::transpose() %>%
    purrr::map(unlist) %>%
    tibble::as_tibble()
}

#' Get Experiment
#'
#' Gets metadata for an experiment and a list of runs for the experiment.
#'
#' @param experiment_id Identifer to get an experiment. Attempts to obtain the active experiment
#'   if not provided.
#' @template roxlate-client
#' @export
mlflow_get_experiment <- function(experiment_id = NULL, client = NULL) {
  client <- client %||% mlflow_client()
  experiment_id <- cast_string(experiment_id)
  response <- mlflow_rest(
    "experiments", "get",
    client = client, query = list(experiment_id = experiment_id)
  )
  response$experiment %>%
    tibble::new_tibble(nrow = 1, class = "tbl_mlflow_experiment")
}

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
mlflow_log_metric <- function(key, value, timestamp = NULL, client = NULL, run_id = NULL) {
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)
  if (!is.numeric(value)) {
    stop(
      "Metric `", key, "`` must be numeric but ", class(value)[[1]], " found.",
      call. = FALSE
    )
  }
  timestamp <- timestamp %||% current_time()
  mlflow_rest("runs", "log-metric", client = client, verb = "POST", data = list(
    run_uuid = run_id,
    key = key,
    value = value,
    timestamp = timestamp
  ))
  invisible(value)
}

#' Delete Experiment
#'
#' Marks an experiment and associated runs, params, metrics, etc. for deletion. If the
#'   experiment uses FileStore, artifacts associated with experiment are also deleted.
#'
#' @param experiment_id ID of the associated experiment. This field is required.
#' @template roxlate-client
#' @export
mlflow_delete_experiment <- function(experiment_id, client = NULL) {
  if (experiment_id == mlflow_active_experiment_id())
    stop("Cannot delete an active experiment.", call. = FALSE)

  client <- client %||% mlflow_client()
  mlflow_rest(
    "experiments", "delete",
    verb = "POST", client = client,
    data = list(experiment_id = experiment_id)
  )
  mlflow_get_experiment(experiment_id)
}

#' Restore Experiment
#'
#' Restores an experiment marked for deletion. This also restores associated metadata,
#'   runs, metrics, and params. If experiment uses FileStore, underlying artifacts
#'   associated with experiment are also restored.
#'
#' Throws `RESOURCE_DOES_NOT_EXIST` if the experiment was never created or was permanently deleted.
#'
#' @param experiment_id ID of the associated experiment. This field is required.
#' @template roxlate-client
#' @export
mlflow_restore_experiment <- function(experiment_id, client = NULL) {
  client <- client %||% mlflow_client()
  mlflow_rest(
    "experiments", "restore",
    client = client, verb = "POST",
    data = list(experiment_id = experiment_id)
  )
  mlflow_get_experiment(experiment_id)
}

#' Rename Experiment
#'
#' Renames an experiment.
#'
#' @template roxlate-client
#' @param experiment_id ID of the associated experiment. This field is required.
#' @param new_name The experiment’s name will be changed to this. The new name must be unique.
#' @export
mlflow_rename_experiment <- function(new_name, experiment_id = NULL, client = NULL) {
  experiment_id <- resolve_experiment_id(experiment_id)

  client <- client %||% mlflow_client()
  mlflow_rest(
    "experiments", "update",
    client = client, verb = "POST",
    data = list(
      experiment_id = experiment_id,
      new_name = new_name
    )
  )
  experiment <- mlflow_get_experiment(experiment_id)

  if (identical(experiment_id, mlflow_active_experiment_id())) {
    # Update active experiment if we rename it
    mlflow_set_active_experiment(experiment)
  }

  experiment
}

#' Create Run
#'
#' reate a new run within an experiment. A run is usually a single execution of a machine learning or data ETL pipeline.
#'
#' MLflow uses runs to track Param, Metric, and RunTag, associated with a single execution.
#'
#' @param experiment_id Unique identifier for the associated experiment.
#' @param user_id User ID or LDAP for the user executing the run.
#' @param run_name Human readable name for run.
#' @param source_type Originating source for this run. One of Notebook, Job, Project, Local or Unknown.
#' @param source_name String descriptor for source. For example, name or description of the notebook, or job name.
#' @param start_time Unix timestamp of when the run started in milliseconds.
#' @param source_version Git version of the source code used to create run.
#' @param entry_point_name Name of the entry point for the run.
#' @param tags Additional metadata for run in key-value pairs.
#' @template roxlate-client
#' @export
mlflow_create_run <- function(experiment_id, user_id = NULL, run_name = NULL, source_type = NULL,
                              source_name = NULL, entry_point_name = NULL, start_time = NULL,
                              source_version = NULL, tags = NULL, client = NULL) {
  client <- client %||% mlflow_client()
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

  parse_run(response$run)
}

#' Delete a Run
#'
#' @template roxlate-client
#' @template roxlate-run-id
#' @export
mlflow_delete_run <- function(run_id, client = NULL) {
  client <- client %||% mlflow_client()
  run_id <- cast_string(run_id)
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
  client <- client %||% mlflow_client()
  run_id <- cast_string(run_id)
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
mlflow_get_run <- function(run_id, client = NULL) {
  client <- client %||% mlflow_client()
  run_id <- cast_string(run_id)
  response <- mlflow_rest(
    "runs", "get",
    client = client, verb = "GET",
    query = list(run_uuid = run_id)
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
                             client = NULL, run_id = NULL) {
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
mlflow_set_tag <- function(key, value, client = NULL, run_id = NULL) {
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)

  key <- cast_string(key)
  value <- cast_string(value)

  mlflow_rest("runs", "set-tag", client = client, verb = "POST", data = list(
    run_uuid = run_id,
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
mlflow_log_param <- function(key, value, client = NULL, run_id = NULL) {
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)

  key <- cast_string(key)
  value <- cast_string(value)

  mlflow_rest("runs", "log-parameter", client = client, verb = "POST", data = list(
    run_uuid = run_id,
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
mlflow_get_metric_history <- function(run_id, metric_key, client = NULL) {
  client <- client %||% mlflow_client()

  run_id <- cast_string(run_id)
  metric_key <- cast_string(metric_key)

  response <- mlflow_rest(
    "metrics", "get-history",
    client = client, verb = "GET",
    query = list(run_uuid = run_id, metric_key = metric_key)
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
#' @param experiment_ids List of experiment IDs to search over.
#' @param filter A filter expression over params, metrics, and tags, allowing returning a subset of runs.
#'   The syntax is a subset of SQL which allows only ANDing together binary operations between a param/metric/tag and a constant.
#' @param run_view_type Run view type.
#'
#' @export
mlflow_search_runs <- function(experiment_ids, filter = NULL,
                               run_view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL"), client = NULL) {
  client <- client %||% mlflow_client()

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
mlflow_list_artifacts <- function(run_id, path = NULL, client = NULL) {
  client <- client %||% mlflow_client()

  response <-   mlflow_rest(
    "artifacts", "list",
    client = client, verb = "GET",
    query = list(
      run_uuid = run_id,
      path = path
    )
  )

  message(glue::glue("Root URI: {uri}", uri = response$root_uri))

  response$files %>%
    purrr::transpose() %>%
    purrr::map(unlist) %>%
    tibble::as_tibble()
}

#' Terminate a Run
#'
#' Terminates a run.
#'
#' @param status Updated status of the run. Defaults to `FINISHED`.
#' @param end_time Unix timestamp of when the run ended in milliseconds.
#' @template roxlate-run-id
#' @template roxlate-client
#' @export
mlflow_set_terminated <- function(run_id, status = c("FINISHED", "SCHEDULED", "FAILED", "KILLED"),
                                  end_time = NULL, client = NULL) {
  client <- client %||% mlflow_client()

  status <- match.arg(status)
  end_time <- end_time %||% current_time()
  response <- mlflow_rest("runs", "update", verb = "POST", client = client, data = list(
    run_uuid = run_id,
    status = status,
    end_time = end_time
  ))
  parse_run_info(response$run_info)
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
mlflow_download_artifacts <- function(run_id, path, client = NULL) {
  client <- client %||% mlflow_client()
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

#' Get Experiment by Name
#'
#' Get meta data for experiment by name.
#'
#' @param name The experiment name.
#' @template roxlate-client
#' @export
mlflow_get_experiment_by_name <- function(name, client = NULL) {
  client <- client %||% mlflow_client()
  exps <- mlflow_list_experiments(client = client)
  if (is.null(exps)) stop("No experiments found.", call. = FALSE)

  experiment <- exps[exps$name == name, ]
  if (nrow(experiment)) experiment else stop(glue::glue("Experiment `{exp}` not found.", exp = name), call. = FALSE)
}

#' List Run Infos
#'
#' List run infos.
#'
#' @param experiment_id Experiment ID.
#' @param run_view_type Run view type.
#' @template roxlate-client
#' @export
mlflow_list_run_infos <- function(experiment_id,
                                  run_view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL"), client = NULL) {
  client <- client %||% mlflow_client()

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
mlflow_log_artifact <- function(path, artifact_path = NULL, client = NULL, run_id = NULL) {
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
