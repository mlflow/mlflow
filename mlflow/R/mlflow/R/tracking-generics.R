#' List Experiments
#'
#' Gets a list of all experiments.
#'
#' @param view_type Qualifier for type of experiments to be returned. Defaults to `ACTIVE_ONLY`.
#' @template roxlate-client
#' @export
mlflow_list_experiments <- function(view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL"), client = NULL) {
  UseMethod("mlflow_list_experiments", client)
}

#' @export
mlflow_list_experiments.default <- function(view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL"), client = NULL) {
  stop("`client` must be an `mlflow_client` object.", call. = FALSE)
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
  UseMethod("mlflow_log_metric", client)
}
