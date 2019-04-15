#' Set Experiment
#'
#' Sets an experiment as the active experiment. If the experiment does not exist,
#'   creates an experiment with provided name.
#'
#' @param experiment_name Name of experiment to be activated.
#' @template roxlate-fluent
#' @export
mlflow_set_experiment <- function(experiment_name) {
  client <- mlflow_client()
  experiment <- tryCatch(
    mlflow_get_experiment_by_name(client = client, name = experiment_name),
    function(e) {
      message("Experiment `", experiment_name, "` does not exist. Creating a new experiment.")
      mlflow_create_experiment(client = client, name = experiment_name)
    }
  )

  mlflow_set_active_experiment(experiment)
}

#' Start Run
#'
#' Starts a new run within an experiment, should be used within a \code{with} block.
#'
#' @param run_uuid If specified, get the run with the specified UUID and log metrics
#'   and params under that run. The run's end time is unset and its status is set to
#'   running, but the run's other attributes remain unchanged.
#' @param experiment_id Used only when `run_uuid` is unspecified. ID of the experiment under
#'   which to create the current run. If unspecified, the run is created under
#'   a new experiment with a randomly generated name.
#' @param source_name Name of the source file or URI of the project to be associated with the run.
#'   Defaults to the current file if none provided.
#' @param source_version Optional Git commit hash to associate with the run.
#' @param entry_point_name Optional name of the entry point for to the current run.
#' @param source_type Integer enum value describing the type of the run  ("local", "project", etc.).
#' @template roxlate-fluent
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
  active_run <- mlflow_get_active_run()
  if (!is.null(active_run)) {
    stop("Run with UUID ", mlflow_get_active_run_id(), " is already active.",
         call. = FALSE)
  }

  existing_run_uuid <- run_uuid %||% {
    env_run_id <- Sys.getenv("MLFLOW_RUN_ID")
    if (nchar(env_run_id)) env_run_id
  }

  client <- mlflow_client()

  run <- if (!is.null(existing_run_uuid)) {
    # This is meant to pick up existing run when we're inside `mlflow_source()` called via `mlflow run`.
    mlflow_get_run(client = client, run_id = existing_run_uuid)
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
  mlflow_set_active_run(run)
}


mlflow_get_run_context <- function(client, ...) {
  UseMethod("mlflow_get_run_context")
}

mlflow_get_run_context.default <- function(client, source_name, source_version, experiment_id,
                                           ...) {
  list(client = client,
       source_name = source_name %||% get_source_name(),
       source_version = source_version %||% get_source_version(),
       experiment_id = experiment_id %||% 0,
       ...)
}

#' End a Run
#'
#' Ends an active MLflow run (if there is one).
#'
#' @param status Updated status of the run. Defaults to `FINISHED`.
#' @template roxlate-fluent
#'
#' @export
mlflow_end_run <- function(status = c("FINISHED", "SCHEDULED", "FAILED", "KILLED")) {
  active_run <- mlflow_get_active_run()
  if (!is.null(active_run)) {
    status <- match.arg(status)
    client <- mlflow_client()
    mlflow_set_terminated(client = client, run_id = mlflow_get_run_id(active_run), status = status)
    mlflow_set_active_run(NULL)
  }
  invisible(NULL)
}
