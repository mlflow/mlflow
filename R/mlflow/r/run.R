#' Run in MLflow
#'
#' Runs the given file, expression or function within
#' the context of an MLflow run.
#'
#' @param x A file, expression or function to run.
#'
#' @export
mlflow_run <- function(x) {
  UseMethod("mlflow_run")
}

#' @export
mlflow_run.character <- function(x) {
}

#' @export
mlflow_run.function <- function(x) {
}
