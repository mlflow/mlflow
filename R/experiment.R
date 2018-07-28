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
#' mlflow_list_experiments()
#'
#' # list experiments in remote MLflow server
#' mlflow_tracking_url("http://tracking-server:5000")
#' mlflow_list_experiments()
#' }
#'
#' @export
mlflow_list_experiments <- function() {
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
#' # create local experiment
#' mlflow_create_experiment("My Experiment")
#'
#' # create experiment in remote MLflow server
#' mlflow_tracking_url("http://tracking-server:5000")
#' mlflow_experiments_create("My Experiment")
#' }
#'
#' @export
mlflow_create_experiment <- function(name) {
  response <- mlflow_rest("experiments", "create", verb = "POST", data = list(name = name))
  response$experimentId
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
#' mlflow_tracking_url("http://tracking-server:5000")
#' mlflow_experiment("My Experiment")
#' }
#'
#' @export
mlflow_experiment <- function(name) {
  if (!name %in% mlflow_list_experiments()$name) {
    mlflow_create_experiment(name)
  }

  exps <- mlflow_list_experiments()
  experiment_id <- exps[exps$name == "Test",]$experiment_id

  Sys.setenv(MLFLOW_EXPERIMENT_ID = experiment_id)

  invisible(experiment_id)
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
