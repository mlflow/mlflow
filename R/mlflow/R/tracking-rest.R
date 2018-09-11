#' List Experiments
#'
#' Retrieves MLflow experiments as a data frame.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' # list local experiments
#' mlflow_list_experiments()
#'
#' # list experiments in remote MLflow server
#' mlflow_set_tracking_uri("http://tracking-server:5000")
#' mlflow_list_experiments()
#' }
#'
#' @export
mlflow_list_experiments <- function() {
  response <- mlflow_rest("experiments", "list")
  exps <- response$experiments

  exps$artifact_location <- mlflow_relative_paths(exps$artifact_location)
  exps
}

#' Create Experiment
#'
#' Creates an MLflow experiment.
#'
#' @param name The name of the experiment to create.
#' @param activate Whether to set the created experiment as the active experiment. Defaults to `TRUE`.
#' @param artifact_location Location where all artifacts for this experiment are stored. If
#'   not provided, the remote server will select an appropriate default.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' # create local experiment
#' mlflow_create_experiment("My Experiment")
#'
#' # create experiment in remote MLflow server
#' mlflow_set_tracking_uri("http://tracking-server:5000")
#' mlflow_create_experiment("My Experiment")
#' }
#'
#' @export
mlflow_create_experiment <- function(name, artifact_location = NULL, activate = TRUE) {
  experiments <- mlflow_list_experiments()
  experiment_id <- if (name %in% experiments$name) {
    message("Experiment with name \"", name, "\" already exists.")
    experiments[experiments$name == name, ]$experiment_id
  } else {
    response <- mlflow_rest(
      "experiments", "create", verb = "POST",
      data = list(name = name, artifact_location = artifact_location)
    )
    response$experiment_id
  }

  if (activate) mlflow_set_active_experiment(experiment_id)
  invisible(experiment_id)
}

#' Get Experiment
#'
#' Get meta data for experiment and a list of runs for this experiment.
#'
#' @param experiment_id Identifer to get an experiment.
#' @export
mlflow_get_experiment <- function(experiment_id) {
  response <- mlflow_rest("experiments", "get", query = list(experiment_id = experiment_id))
  response
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
#' @param status Current status of the run. One of RUNNING, SCHEDULE, FINISHED, FAILED, KILLED.
#' @param start_time Unix timestamp of when the run started in milliseconds.
#' @param end_time Unix timestamp of when the run ended in milliseconds.
#' @param source_version Git version of the source code used to create run.
#' @param entry_point_name Name of the entry point for the run.
#' @param tags Additional metadata for run in key-value pairs.
#' @export
mlflow_create_run <- function(user_id = mlflow_user(),
                              run_name = NULL, source_type = NULL, source_name = NULL,
                              status = NULL, start_time = NULL, end_time = NULL,
                              source_version = NULL, entry_point_name = NULL,
                              tags = NULL, experiment_id = NULL) {
  experiment_id <- experiment_id %||% mlflow_active_experiment()
  start_time <- start_time %||% current_time()

  tags <- if (!is.null(tags)) tags %>%
    purrr::imap(~ list(key = .y, value = .x)) %>%
    unname()

  response <- mlflow_rest("runs", "create", verb = "POST", data = list(
    experiment_id = experiment_id,
    user_id = user_id,
    run_name = run_name,
    source_type = source_type,
    source_name = source_name,
    status = status,
    start_time = start_time,
    end_time = end_time,
    source_version = source_version,
    entry_point_name = entry_point_name,
    tags = tags
  ))

  tidy_run_info(response$run$info)
}

#' Get Run
#'
#' Get meta data, params, tags, and metrics for run. Only last logged value for each metric is returned.
#'
#' @param run_uuid Unique ID for the run.
#'
#' @export
mlflow_get_run <- function(run_uuid) {
  response <- mlflow_rest("runs", "get", query = list(run_uuid = run_uuid))
  run <- purrr::compact(response$run)
  run %>%
    purrr::map_at("info", tidy_run_info)
}

#' Log Metric
#'
#' API to log a metric for a run. Metrics key-value pair that record a single float measure.
#'   During a single execution of a run, a particular metric can be logged several times.
#'   Backend will keep track of historical values along with timestamps.
#'
#' @param run_uuid Unique ID for the run.
#' @param key Name of the metric.
#' @param value Float value for the metric being logged.
#' @param timestamp Unix timestamp in milliseconds at the time metric was logged.
#' @export
mlflow_log_metric <- function(key, value, timestamp = NULL, run_uuid = NULL) {
  if (!rlang::inherits_any(value, c("character", "numeric", "integer"))) {
    stop("Metric ", key, " must be a character or numeric but ", class(value), " found.")
  }

  run_uuid <- mlflow_ensure_run_id(run_uuid)
  timestamp <- timestamp %||% current_time()
  response <- mlflow_rest("runs", "log-metric", verb = "POST", data = list(
    run_uuid = run_uuid,
    key = key,
    value = value,
    timestamp = timestamp
  ))

  invisible(value)
}

#' Log Parameter
#'
#' API to log a parameter used for this run. Examples are params and hyperparams
#'   used for ML training, or constant dates and values used in an ETL pipeline.
#'   A params is a STRING key-value pair. For a run, a single parameter is allowed
#'   to be logged only once.
#'
#' @param run_uuid Unique ID for the run for which parameter is recorded.
#' @param key Name of the parameter.
#' @param value String value of the parameter.
#' @export
mlflow_log_param <- function(key, value, run_uuid = NULL) {
  run_uuid <- mlflow_ensure_run_id(run_uuid)
  response <- mlflow_rest("runs", "log-parameter", verb = "POST", data = list(
    run_uuid = run_uuid,
    key = key,
    value = as.character(value)
  ))

  invisible(value)
}

#' Get Param
#'
#' Get a param value.
#'
#' @return The param value as a named list.
#' @param run_uuid ID of the run from which to retrieve the param value.
#' @param param_name Name of the param. This field is required.
#' @export
mlflow_get_param <- function(param_name, run_uuid = NULL) {
  mlflow_get_or_create_active_connection()
  run_uuid <- run_uuid %||%
    mlflow_active_run()$run_info$run_uuid %||%
    stop("`run_uuid` must be specified when there is no active run.")

  response <- mlflow_rest("params", "get", query = list(
    run_uuid = run_uuid,
    param_name = param_name
  ))

  as.data.frame(response$parameter, stringsAsFactors = FALSE)
}

#' Get Metric
#'
#' API to retrieve the logged value for a metric during a run. For a run, if this
#'   metric is logged more than once, this API will retrieve only the latest value logged.
#'
#' @param run_uuid Unique ID for the run for which metric is recorded.
#' @param metric_key Name of the metric.
#' @export
mlflow_get_metric <- function(metric_key, run_uuid = NULL) {
  mlflow_get_or_create_active_connection()
  run_uuid <- run_uuid %||%
    mlflow_active_run()$run_info$run_uuid %||%
    stop("`run_uuid` must be specified when there is no active run.")

  response <- mlflow_rest("metrics", "get", query = list(
    run_uuid = run_uuid,
    metric_key = metric_key
  ))

  metric <- response$metric
  metric$timestamp <- as.POSIXct(as.double(metric$timestamp) / 1000, origin = "1970-01-01")
  as.data.frame(metric, stringsAsFactors = FALSE)
}

#' Get Metric History
#'
#' For cases that a metric is logged more than once during a run, this API can be used
#'   to retrieve all logged values for this metric.
#'
#' @param run_uuid Unique ID for the run for which metric is recorded.
#' @param metric_key Name of the metric.
#' @export
mlflow_get_metric_history <- function(metric_key, run_uuid = NULL) {
  mlflow_get_or_create_active_connection()
  run_uuid <- run_uuid %||%
    mlflow_active_run()$run_info$run_uuid %||%
    stop("`run_uuid` must be specified when there is no active run.")

  response <- mlflow_rest("metrics", "get-history", query = list(
    run_uuid = run_uuid,
    metric_key = metric_key
  ))

  metrics <- response$metrics
  metrics$timestamp <- as.POSIXct(as.double(metrics$timestamp) / 1000, origin = "1970-01-01")
  as.data.frame(metrics, stringsAsFactors = FALSE)
}

#' Update Run
#'
#' @param run_uuid Unique identifier for the run.
#' @param status Updated status of the run. Defaults to `FINISHED`.
#' @param end_time Unix timestamp of when the run ended in milliseconds.
#' @export
mlflow_update_run <- function(status = c("FINISHED", "SCHEDULED", "FAILED", "KILLED"),
                              end_time = NULL,
                              run_uuid = NULL) {
  mlflow_get_or_create_active_connection()
  run_uuid <- run_uuid %||%
    mlflow_active_run()$run_info$run_uuid %||%
    stop("`run_uuid` must be specified when there is no active run.")

  status <- match.arg(status)
  end_time <- end_time %||% current_time()

  response <- mlflow_rest("runs", "update", verb = "POST", data = list(
    run_uuid = run_uuid,
    status = status,
    end_time = end_time
  ))

  tidy_run_info(response$run_info)
}

current_time <- function() {
  round(as.numeric(Sys.time()) * 1000)
}

milliseconds_to_date <- function(x) as.POSIXct(as.double(x) / 1000, origin = "1970-01-01")

tidy_run_info <- function(run_info) {
  df <- as.data.frame(run_info, stringsAsFactors = FALSE)
  df$start_time <- milliseconds_to_date(df$start_time %||% NA)
  df$end_time <- milliseconds_to_date(df$end_time %||% NA)
  df
}
