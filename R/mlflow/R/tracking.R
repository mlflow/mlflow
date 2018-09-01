#' Active Experiment
#'
#' Retrieve or set the active experiment.
#'
#' @name active_experiment
#' @export
mlflow_active_experiment <- function() {
  .globals$active_experiment
}

#' @rdname active_experiment
#' @param experiment_id Identifer to get an experiment.
#' @export
mlflow_set_active_experiment <- function(experiment_id) {
  if (!identical(experiment_id, .globals$active_experiment)) {
    .globals$active_experiment <- experiment_id
    mlflow_set_active_run(NULL)
  }

  invisible(experiment_id)
}

#' Active Run
#'
#' Retrieves or sets the active run.
#'
#' @name active_run
#' @export
mlflow_active_run <- function() {
  .globals$active_run
}

#' @rdname active_run
#' @param run The run object to make active.
#' @export
mlflow_set_active_run <- function(run) {
  .globals$active_run <- run
  invisible(run)
}

mlflow_relative_paths <- function(paths) {
  gsub(paste0("^", file.path(getwd(), "")), "", paths)
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

    mlflow_create_run(
      experiment_id = experiment_id,
      source_name = source_name %||% get_source_name(),
      source_version = source_version %||% get_source_version(),
      entry_point_name = entry_point_name,
      source_type = source_type
    )
  }

  new_mlflow_active_run(run_info)
}

get_executing_file_name <- function() {
  pattern <- "^--file="
  v <- grep(pattern, commandArgs(), value = TRUE)
  file_name <- gsub(pattern, "", v)
  if (length(file_name)) file_name
}

get_source_name <- function() {
  get_executing_file_name() %||% "<console>"
}

get_source_version <- function() {
  file_name <- get_executing_file_name()
  tryCatch(
    error = function(cnd) NULL,
    {
      repo <- git2r::repository(file_name, discover = TRUE)
      commit <- git2r::commits(repo, n = 1)
      commit[[1]]@sha
    }
  )
}

#' End Run
#'
#' End the active run.
#'
#' @param status Ending status of the run, defaults to `FINISHED`.
#' @export
mlflow_end_run <- function(status = "FINISHED") {
  if (!is.null(mlflow_active_run())) {
    mlflow_update_run(status = status)
    mlflow_set_active_run(NULL)
  }
  invisible(NULL)
}

new_mlflow_active_run <- function(run_info) {
  run <- structure(
    list(run_info = run_info),
    class = c("mlflow_active_run")
  )
  mlflow_set_active_run(run)
  run
}

#' @export
with.mlflow_active_run <- function(data, expr, ...) {
  runid <- as.character(data$run_info$run_uuid)

  tryCatch(
    error = function(cnd) {
      message(cnd)
      mlflow_update_run(run_uuid = runid, status = "FAILED", end_time = current_time())
    },
    interrupt = function(cnd) mlflow_update_run(
      run_uuid = runid, status = "KILLED", end_time = current_time()
    ),
    {
      force(expr)
      mlflow_end_run()
    }
  )

  invisible(NULL)
}

mlflow_ensure_run_id <- function(run_uuid) {
  if (is.null(run_uuid)) {
    run <- mlflow_active_run() %||% mlflow_start_run()
    run$run_info$run_uuid
  } else {
    mlflow_get_or_create_active_connection()
    tryCatch(mlflow_get_run(run_uuid)$info$run_uuid, error = function(e) {
      stop(
        "Run \"", run_uuid, "\" cannot be found in current tracking server at ",
        mlflow_tracking_uri(), "."
      )
    })
  }
}

#' Log Model
#'
#' Logs a model in the given run. Similar to `mlflow_save_model()`
#' but stores model as an artifact within the active run.
#'
#' @param fn The serving function that will perform a prediction.
#' @param artifact_path Destination path where this MLflow compatible model
#'   will be saved.
#' @param run_uuid The run associated with the model to be logged.
#'
#' @export
mlflow_log_model <- function(fn, artifact_path, run_uuid = NULL) {
  temp_path <- fs::path_temp(artifact_path)
  mlflow_save_model(fn, path = temp_path)
  mlflow_log_artifact(path = temp_path, artifact_path = artifact_path, run_uuid = run_uuid)
}
