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
