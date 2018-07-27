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
#' mlflow_experiments()
#'
#' # list experiments in remote MLflow server
#' mlflow_tracking_url("http://tracking-server:5000")
#' mlflow_experiments()
#' }
#'
#' @export
mlflow_experiments <- function() {
  response <- mlflow_rest("experiments", "list")
  exps <- response$experiments

  exps$artifact_location <- mlflow_relative_paths(exps$artifact_location)
  exps
}

mlflow_experiments_rest <- function() {

}

#' Create Experiment
#'
#' Creates an MLflow experiment.
#'
#' @param name The name of the experiment to create.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' # list local experiments
#' mlflow_experiment_create()
#'
#' # create experiment in remote MLflow server
#' mlflow_tracking_url("http://tracking-server:5000")
#' mlflow_experiments_create("My Experiment")
#' }
#'
#' @export
mlflow_experiments_create <- function(name) {
  response <- mlflow_rest("experiments", "create", verb = "POST", data = list(name = name))
  response$experimentId
}

mlflow_relative_paths <- function(paths) {
  gsub(paste0("^", file.path(getwd(), "")), "", paths)
}

#' Log to MLflow
#'
#' Logs a value to MLflow for the active run.
#'
#' @param name The name to identify this log entry.
#' @param value The value to log into this entry.
#'
#' @export
mlflow_log <- function(name, value) {
  invisible(NULL)
}
