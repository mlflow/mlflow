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

#' Active Experiment
#'
#' Retrieve or set the active experiment.
#'
#' @name active_experiment
#' @export
mlflow_get_active_experiment_id <- function() {
  .globals$active_experiment_id
}

#' @rdname active_experiment
#' @param experiment_id Identifer to get an experiment.
#' @export
mlflow_set_active_experiment_id <- function(experiment_id) {
  if (!identical(experiment_id, .globals$active_experiment_id)) {
    .globals$active_experiment_id <- experiment_id
  }

  invisible(experiment_id)
}
