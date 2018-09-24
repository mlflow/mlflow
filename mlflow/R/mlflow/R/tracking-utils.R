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

new_mlflow_active_run <- function(run_info) {
  run <- structure(
    list(run_info = run_info),
    class = c("mlflow_active_run")
  )
  mlflow_set_active_run(run)
  run
}

mlflow_get_or_start_run <- function() {
  mlflow_active_run() %||% mlflow_start_run()
}

#' @export
with.mlflow_active_run <- function(data, expr, ...) {

  tryCatch(
    {
      force(expr)
      mlflow_end_run()
    },
    error = function(cnd) {
      message(cnd)
      mlflow_end_run(status = "FAILED")
    },
    interrupt = function(cnd) mlflow_update_run(status = "KILLED")
  )

  invisible(NULL)
}
