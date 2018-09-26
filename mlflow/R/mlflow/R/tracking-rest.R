mlflow_rest_path <- function(version) {
  switch(
    version,
    "2.0" = "ajax-api/2.0/preview/mlflow"
  )
}

mlflow_rest_body <- function(data) {
  data <- Filter(length, data)
  paste0(
    "\"",
    gsub(
      "\\\"",
      "\\\\\"",
      as.character(
        jsonlite::toJSON(data, auto_unbox = TRUE)
      )
    ),
    "\""
  )
}

#' @importFrom httr add_headers
mlflow_rest_headers <- function() {
  add_headers("Content-Type" = "application/json")
}

#' @importFrom httr timeout
mlflow_rest_timeout <- function() {
  timeout(getOption("mlflow.rest.timeout", 1))
}

#' @importFrom httr content
#' @importFrom httr GET
#' @importFrom httr POST
#' @importFrom jsonlite fromJSON
#' @importFrom xml2 as_list
mlflow_rest <- function(..., client, query = NULL, data = NULL, verb = "GET", version = "2.0") {
  args <- list(...)
  tracking_url <- client$server_url

  api_url <- file.path(
    tracking_url,
    mlflow_rest_path(version),
    paste(args, collapse = "/")
  )

  response <- switch(
    verb,
    GET = GET(api_url, query = query, mlflow_rest_timeout()),
    POST = POST(
      api_url,
      body = mlflow_rest_body(data),
      mlflow_rest_headers(),
      mlflow_rest_timeout()
    ),
    stop("Verb '", verb, "' is unsupported.")
  )

  if (identical(response$status_code, 500L)) {
    stop(as_list(content(response))$html$body$p[[1]])
  }

  text <- content(response, "text", encoding = "UTF-8")
  jsonlite::fromJSON(text)
}

#' Create Experiment - Tracking Client
#'
#' Creates an MLflow experiment.
#'
#' @param name The name of the experiment to create.
#' @param artifact_location Location where all artifacts for this experiment are stored. If
#'   not provided, the remote server will select an appropriate default.
#' @param client
#'
#' @export
mlflow_client_create_experiment <- function(client, name, artifact_location = NULL) {
  name <- forge::cast_string(name)
  response <- mlflow_rest(
    "experiments", "create", client = client, verb = "POST",
    data = list(
      name = name,
      artifact_location = artifact_location
    )
  )
  invisible(response$experiment_id)
}

#' List Experiments
#'
#' Get a list of all experiments.
#'
#' @param view_type Qualifier for type of experiments to be returned. Defaults to `ACTIVE_ONLY`.
#' @template roxlate-client
#' @export
mlflow_client_list_experiments <- function(client, view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL")) {
  view_type <- match.arg(view_type)
  response <- mlflow_rest(
    "experiments", "list", client = client, verb = "GET",
    query = list(
      view_type = view_type
    ))
  exps <- response$experiments

  exps$artifact_location <- mlflow_relative_paths(exps$artifact_location)
  exps
}

#' Get Experiment
#'
#' Get meta data for experiment and a list of runs for this experiment.
#'
#' @param experiment_id Identifer to get an experiment.
#' @template roxlate-client
#' @export
mlflow_client_get_experiment <- function(client, experiment_id) {
  mlflow_rest(
    "experiments", "get", client = client,
    query = list(experiment_id = experiment_id)
  )
}

#' Get Experiment by Name
#'
#' Get meta data for experiment by name.
#'
#' @param name The experiment name.
#' @template roxlate-client
#' @export
mlflow_client_get_experiment_by_name <- function(client, name) {
  exps <- mlflow_list_experiments(client = client)
  experiment <- exps[exps$name == name, ]
  if (nrow(experiment)) experiment else NULL
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
mlflow_client_create_run <- function(
  client, experiment_id, user_id, run_name, source_type,
  source_name, entry_point_name, start_time, source_version, tags
) {
  tags <- if (!is.null(tags)) tags %>%
    purrr::imap(~ list(key = .y, value = .x)) %>%
    unname()

  start_time <- start_time %||% current_time()

  response <- mlflow_rest(
    "runs", "create", client = client, verb = "POST",
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
  new_mlflow_entities_run(response)
}

mlflow_client_update_run <- function(client, run_uuid, status, end_time) {
  mlflow_rest("runs", "update", client = client, verb = "POST", data = list(
    run_uuid = run_uuid,
    status = status,
    end_time = end_time
  ))
}

#' Delete Experiment
#'
#' Mark an experiment and associated runs, params, metrics, … etc for deletion. If the
#'   experiment uses FileStore, artifacts associated with experiment are also deleted.
#'
#' @param experiment_id ID of the associated experiment This field is required.
#' @template roxlate-client
#' @export
mlflow_client_delete_experiment <- function(client, experiment_id) {
  mlflow_rest(
    "experiments", "delete", client = client, verb = "POST",
    data = list(experiment_id = experiment_id),
  )
}

#' Restore Experiment
#'
#' Restore an experiment marked for deletion. This also restores associated metadata,
#'   runs, metrics, and params. If experiment uses FileStore, underlying artifacts
#'   associated with experiment are also restored.
#'
#' Throws RESOURCE_DOES_NOT_EXIST if experiment was never created or was permanently deleted.
#'
#' @param experiment_id ID of the associated experiment This field is required.
#' @template roxlate-client
#' @export
mlflow_client_restore_experiment <- function(client, experiment_id) {
  mlflow_rest(
    "experiments", "restore", client = client, verb = "POST",
    data = list(experiment_id = experiment_id),
  )
}

#' Get Run
#'
#' Get meta data, params, tags, and metrics for run. Only last logged value for each metric is returned.
#'
#' @param run_uuid Unique ID for the run.
#' @template roxlate-client
#'
#' @export
mlflow_client_get_run <- function(client, run_uuid) {
  response <- mlflow_rest(
    "runs", "get", client = client, verb = "GET",
    query = list(run_uuid = run_uuid),
  )
  new_mlflow_entities_run(response)
}

#' Log Metric
#'
#' API to log a metric for a run. Metrics key-value pair that record a single float measure.
#'   During a single execution of a run, a particular metric can be logged several times.
#'   Backend will keep track of historical values along with timestamps.
#'
#' @param key Name of the metric.
#' @param value Float value for the metric being logged.
#' @param timestamp Unix timestamp in milliseconds at the time metric was logged.
#' @export
mlflow_client_log_metric <- function(client, run_uuid, key, value, timestamp = NULL) {
  if (!is.numeric(value)) stop(
    "Metric `", key, "`` must be numeric but ", class(value)[[1]], " found.",
    call. = FALSE
  )
  timestamp <- timestamp %||% current_time()
  mlflow_rest("runs", "log-metric", client = client, verb = "POST", data = list(
    run_uuid = run_uuid,
    key = key,
    value = value,
    timestamp = timestamp
  ))
}

mlflow_client_set_tag <- function(client, run_uuid, key, value) {
  mlflow_rest("runs", "set-tag", client = client, verb = "POST", data = list(
    run_uuid = run_uuid,
    key = key,
    value = value
  ))
}

mlflow_client_log_param <- function(client, run_uuid, key, value) {
  mlflow_rest("runs", "log-parameter", client = client, verb = "POST", data = list(
    run_uuid = run_uuid,
    key = key,
    value = cast_string(value)
  ))
}

mlflow_client_get_param <- function(client, run_uuid, param_name) {
  mlflow_rest("params", "get", client = client, query = list(
    run_uuid = run_uuid,
    param_name = param_name
  ))
}

mlflow_client_get_metric <- function(client, run_uuid, metric_key) {
  mlflow_rest("metrics", "get", client = client, query = list(
    run_uuid = run_uuid,
    metric_key = metric_key
  ))
}

mlflow_client_get_metric_history <- function(client, run_uuid, metric_key) {
  mlflow_rest("metrics", "get-history", client = client, query = list(
    run_uuid = run_uuid,
    metric_key = metric_key
  ))
}
