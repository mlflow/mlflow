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
  c(client, run_id) %<-% resolve_client_and_run_id(client, run_id)
  if (!is.numeric(value)) {
    stop(
      "Metric `", key, "`` must be numeric but ", class(value)[[1]], " found.",
      call. = FALSE
    )
  }
  timestamp <- timestamp %||% current_time()
  mlflow_client_log_metric(client, run_id, key, value, timestamp)
  invisible(value)
}

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
  UseMethod("mlflow_create_experiment", client)
}

#' Get Experiment
#'
#' Gets metadata for an experiment and a list of runs for the experiment.
#'
#' @param experiment_id Identifer to get an experiment.
#' @template roxlate-client
#' @export
mlflow_get_experiment <- function(experiment_id, client = NULL) {
  UseMethod("mlflow_get_experiment", client)
}
