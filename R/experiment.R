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
#' mlflow_experiments(mc)
#'
#' # list experiments in remote MLflow server
#' mlflow_tracking_url("http://tracking-server:5000")
#' mlflow_experiments()
#' }
#'
#' @export
mlflow_experiments <- function() {
  response <- mlflow_rest("experiments", "list")
  response$experiments
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
#' mlflow_experiment_create(mc)
#'
#' # create experiment in remote MLflow server
#' mlflow_tracking_url("http://tracking-server:5000")
#' mlflow_experiments_create("My Experiment")
#' }
#'
#' @export
mlflow_experiments_create <- function(name) {
  response <- mlflow_rest("experiments", "create", verb = "POST", data = list(name = name))
  response
}
