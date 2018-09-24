#' @export
mlflow_create_experiment.NULL <- function(name, artifact_location = NULL, client = NULL) {
  client <- mlflow_client()
  experiment_id <- mlflow_create_experiment.mlflow_client(name, artifact_location, client)
  mlflow_set_active_experiment(experiment_id)
  invisible(experiment_id)
}

#' Start Run
#'
#' Starts a new run within an experiment, should be used within a \code{with} block.
#'
#' @param run_uuid If specified, get the run with the specified UUID and log metrics
#'   and params under that run. The run's end time is unset and its status is set to
#'   running, but the run's other attributes remain unchanged.
#' @param experiment_id Used only when ``run_uuid`` is unspecified. ID of the experiment under
#'   which to create the current run. If unspecified, the run is created under
#'   a new experiment with a randomly generated name.
#' @param source_name Name of the source file or URI of the project to be associated with the run.
#'   Defaults to the current file if none provided.
#' @param source_version Optional Git commit hash to associate with the run.
#' @param entry_point_name Optional name of the entry point for to the current run.
#' @param source_type Integer enum value describing the type of the run  ("local", "project", etc.).
#'
#' @examples
#' \dontrun{
#' with(mlflow_start_run(), {
#'   mlflow_log("test", 10)
#' })
#' }
#'
#' @export
mlflow_start_run <- function(run_uuid = NULL, experiment_id = NULL, source_name = NULL,
                             source_version = NULL, entry_point_name = NULL,
                             source_type = "LOCAL") {
  active_run <- mlflow_active_run()
  if (!is.null(active_run)) {
    stop("Run with UUID ", active_run$run_info$run_uuid, " is already active.",
         call. = FALSE)
  }

  existing_run_uuid <- run_uuid %||% {
    env_run_id <- Sys.getenv("MLFLOW_RUN_ID")
    if (nchar(env_run_id)) env_run_id
  }

  run_info <- if (!is.null(existing_run_uuid)) {
    mlflow_get_run(existing_run_uuid)$info
  } else {
    experiment_id <- as.integer(
      experiment_id %||% mlflow_active_experiment() %||% Sys.getenv("MLFLOW_EXPERIMENT_ID", unset = "0")
    )

    client <- mlflow_client()

    mlflow_create_run(
      client = client,
      experiment_id = experiment_id,
      source_name = source_name %||% get_source_name(),
      source_version = source_version %||% get_source_version(),
      entry_point_name = entry_point_name,
      source_type = source_type
    )
  }

  new_mlflow_active_run(run_info)
}

#' @rdname mlflow_log_metric
#' @export
mlflow_log_metric.NULL <- function(key, value, timestamp = NULL, client = NULL) {
  client <- mlflow_client()
  active_run <- mlflow_active_run()
  run_id <- as.character(active_run$run_info$run_uuid)
  mlflow_log_metric.mlflow_client(
    client = client, key = key, value = value, timestamp = timestamp,
    run_id = run_id
  )
  invisible(value)
}

#' @rdname mlflow_set_tag
#' @export
mlflow_set_tag.NULL <- function(key, value, client = NULL, ...) {
  client <- mlflow_client()
  active_run <- mlflow_active_run()
  run_id <- as.character(active_run$run_info$run_uuid)
  mlflow_set_tag.mlflow_client(
    key = key, value = value, client = client, run_id = run_id
  )
}

#' End a Run
#'
#' End an active MLflow run (if there is one).
#'
#' @param status Updated status of the run. Defaults to `FINISHED`.
#' @export
mlflow_end_run <- function(status = c("FINISHED", "SCHEDULED", "FAILED", "KILLED")) {
  active_run <- mlflow_active_run()
  if (!is.null(active_run)) {
    client <- mlflow_client()
    run_id <- as.character(active_run$run_info$run_uuid)
    mlflow_set_terminated(run_id, status, client = client)
    mlflow_set_active_run(NULL)
  }
  invisible(NULL)
}

#' @rdname mlflow_log_param
#' @export
mlflow_log_param.NULL <- function(key, value, client = NULL, ...) {
  client <- mlflow_client()
  active_run <- mlflow_active_run()
  run_id <- as.character(active_run$run_info$run_uuid)
  mlflow_log_param.mlflow_client(key, value, client, run_id)
  invisible(value)
}


#' @rdname mlflow_log_artifact
#' @export
mlflow_log_artifact.NULL <- function(path, artifact_path = NULL, client = NULL, ...) {
  client <- mlflow_client()
  active_run <- mlflow_active_run()
  run_id <- as.character(active_run$run_info$run_uuid)
  mlflow_log_artifact.mlflow_client(path, artifact_path, client, run_id)
}
