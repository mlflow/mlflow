#' Active Experiment
#'
#' Retrieves the active experiment. An experiment is made active by calling
#' \code{mlflow_experiment()}.
#'
#' @export
mlflow_active_experiment <- function() {
  .globals$active_experiment
}

#' Active Run
#'
#' Retrieves the active run. A run is made active by calling
#' \code{mlflow_create_run()}.
#'
#' @export
mlflow_active_run <- function() {
  .globals$active_run
}

mlflow_set_active_run <- function(run) {
  .globals$active_run <- run
}

mlflow_relative_paths <- function(paths) {
  gsub(paste0("^", file.path(getwd(), "")), "", paths)
}

#' Active Experiment
#'
#' Creates an MLflow experiment and makes it active.
#'
#' @param name The name of the experiment to create.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' # activates experiment
#' mlflow_experiment("My Experiment")
#'
#' # activates experiment in remote MLflow server
#' mlflow_set_tracking_uri("http://tracking-server:5000")
#' mlflow_experiment("My Experiment")
#' }
#'
#' @export
mlflow_experiment <- function(name) {
  if (!name %in% mlflow_list_experiments()$name) {
    mlflow_create_experiment(name)
  }

  exps <- mlflow_list_experiments()
  experiment_id <- exps[exps$name == name,]$experiment_id

  .globals$active_experiment <- experiment_id

  invisible(experiment_id)
}

#' Start Run
#'
#' Starts a new run within an experiment, should be used within a \code{with} block.
#'
#' @inheritParams mlflow_create_run
#'
#' @examples
#' \dontrun{
#' with(mlflow_start_run(), {
#'   mlflow_log("test", 10)
#' })
#' }
#'
#' @export
mlflow_start_run <- function(user_id = NULL,
                             run_name = NULL, source_type = NULL, source_name = NULL,
                             status = NULL, start_time = NULL, end_time = NULL,
                             source_version = NULL, artifact_uri = NULL, entry_point_name = NULL,
                             run_tags = NULL, experiment_id = NULL) {
  experiment_id <- experiment_id %||% mlflow_active_experiment()
  run_info <- mlflow_create_run(
    experiment_id = experiment_id,
    user_id = user_id,
    run_name = run_name,
    source_type = source_type,
    source_name = source_name,
    status = status,
    start_time = start_time,
    end_time = end_time,
    source_version = source_version,
    artifact_uri = artifact_uri,
    entry_point_name = entry_point_name,
    run_tags = run_tags
  )

  new_mlflow_active_run(run_info)
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
  structure(
    list(run_info = run_info),
    class = c("mlflow_active_run")
  )
}

#' @export
with.mlflow_active_run <- function(x, code) {
  runid <- as.character(x$run_info$run_uuid)

  tryCatch(
    error = function(cnd) {
      message(cnd)
      mlflow_update_run(run_uuid = runid,status = "FAILED", end_time = current_time())
    },
    interrupt = function(cnd) mlflow_update_run(
      run_uuid = runid, status = "KILLED", end_time = current_time()
    ),
    {
      force(code)
      mlflow_end_run()
    }
  )

  invisible(NULL)
}

mlflow_ensure_run <- function(run_uuid) {
  if (is.null(run_uuid)) {
    mlflow_active_run()$run_info$run_uuid %||% mlflow_start_run()$run_info$run_uuid
  } else {
    mlflow_start_run(run_uuid)$run_info$run_uuid
  }
}

#' Log Artifact
#'
#' Logs an specific file or directory as an artifact.
#'
#' @param path The file or directory to log as an artifact.
#'
#' @export
mlflow_log_artifact <- function(path) {
  stop("Not implemented.")
}

#' Log Model
#'
#' Logs a model in the given run. Similar to `mlflow_save_model()`
#' but stores model only as an artifact within the active run.
#'
#' @param f The serving function that will perform a prediction.
#' @param path Destination path where this MLflow compatible model
#'   will be saved.
#'
#' @export
mlflow_log_model <- function(f, path = "model") {
  stop("Not implemented.")
}
