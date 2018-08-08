#' Run Script
#'
#' Runs an R script in the context of 'MLflow', designed to be used by
#' tools like terminal or RStudio.
#'
#' @param script Path to the script to be run.
#' @export
mlflow_script <- function(script) {
  source(script)
}
