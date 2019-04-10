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
  response <- mlflow_client_create_experiment(client, name, artifact_location)
  # TODO: return more info here?
  invisible(response$experiment_id)
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
  response <- mlflow_client_list_experiments(client = client, view_type = view_type)
  response$experiments %>%
    purrr::map(new_mlflow_rest_data_experiment) %>%
    new_mlflow_rest_data_array(type = "Experiment")
}

#' Get Experiment
#'
#' Gets metadata for an experiment and a list of runs for the experiment.
#'
#' @param experiment_id Identifer to get an experiment.
#' @template roxlate-client
#' @export
mlflow_get_experiment <- function(experiment_id, client = NULL) {
  client <- client %||% mlflow_client()
  experiment_id <- cast_string(experiment_id)
  response <- mlflow_client_get_experiment(client = client, experiment_id = experiment_id)
  new_mlflow_rest_data_experiment(response$experiment)
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

#' Delete Experiment
#'
#' Marks an experiment and associated runs, params, metrics, etc. for deletion. If the
#'   experiment uses FileStore, artifacts associated with experiment are also deleted.
#'
#' @param experiment_id ID of the associated experiment. This field is required.
#' @template roxlate-client
#' @export
mlflow_delete_experiment <- function(experiment_id, client = NULL) {
  client <- client %||% mlflow_client()
  mlflow_client_delete_experiment(client = client, experiment_id = experiment_id)
  invisible(NULL)
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
  mlflow_client_restore_experiment(client = client, experiment_id = experiment_id)
  invisible(NULL)
}

#' Rename Experiment
#'
#' Renames an experiment.
#'
#' @param experiment_id ID of the associated experiment. This field is required.
#' @param new_name The experiment’s name will be changed to this. The new name must be unique.
#' @export
mlflow_rename_experiment <- function(experiment_id, new_name, client = NULL) {
  client <- client %||% mlflow_client()
  mlflow_client_rename_experiment(
    client = client, experiment_id = experiment_id,
    new_name = new_name
  )
  invisible(NULL)
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

  response <- mlflow_client_create_run(
    client = client,
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

  # TODO: use REST data structure for consistency?
  new_mlflow_entities_run(response)
}

#' Delete a Run
#'
#' @template roxlate-client
#' @template roxlate-run-id
#' @export
mlflow_delete_run <- function(run_id, client = NULL) {
  client <- client %||% mlflow_client()
  run_id <- cast_string(run_id)
  mlflow_client_delete_run(client = client, run_id = run_id)
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
  mlflow_client_restore_run(client = client, run_id = run_id)
  invisible(NULL)
}
