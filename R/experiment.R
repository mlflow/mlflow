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
  exps <- mlflow_choose_api(mlflow_experiments_cli, mlflow_experiments_rest)

  exps$artifact_location <- mlflow_relative_paths(exps$artifact_location)
  exps
}

mlflow_experiments_rest <- function() {
  response <- mlflow_rest("experiments", "list")
  response$experiments
}

mlflow_experiments_cli <- function() {
  result <- mlflow_cli("experiments", "list", echo = FALSE)
  exps <- read.table(mlflow_cli_file_output(result), skip = 2)
  colnames(exps) <- c(
    "experiment_id",
    "name",
    "artifact_location"
  )

  exps
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
  mlflow_choose_api(mlflow_experiments_create_cli, mlflow_experiments_create_rest, name)
}

mlflow_experiments_create_rest <- function(name) {
  response <- mlflow_rest("experiments", "create", verb = "POST", data = list(name = name))
  response$experimentId
}

mlflow_experiments_create_cli <- function(name) {
  response <- mlflow_cli("experiments", "create", name, echo = FALSE)

  experiment_id_match <- regexec("with id ([0-9]+)", response$stdout)
  regmatches(response$stdout, experiment_id_match)[[1]][[2]]
}

mlflow_relative_paths <- function(paths) {
  gsub(paste0("^", file.path(getwd(), "")), "", paths)
}
