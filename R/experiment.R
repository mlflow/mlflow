#' List Experiments
#'
#' Retrieves MLflow experiments as a data frame.
#'
#' @param mc The MLflow connection created using \code{mlflow_connect()}.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' mc <- mlflow_connect()
#' mlflow_experiments(mc)
#' }
#'
#' @export
mlflow_experiments <- function(mc) {
  response <- mlflow_rest(mc, "experiments", "list")
  response$experiments
}

#' Create Experiment
#'
#' Creates an MLflow experiment.
#'
#' @param mc The MLflow connection created using \code{mlflow_connect()}.
#' @param name The name of the experiment to create.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' mc <- mlflow_connect()
#' mlflow_experiments(mc)
#' }
#'
#' @export
mlflow_experiment_create <- function(mc, name) {
  response <- mlflow_rest(mc, "experiments", "create", verb = "POST", data = list(name = name))
  response
}
