#' Create Experiment
#'
#' Creates an MLflow experiment.
#'
#' @param name The name of the experiment to create.
#' @param artifact_location Location where all artifacts for this experiment are stored. If
#'   not provided, the remote server will select an appropriate default.
#'
#' @export
mlflow_create_experiment <- function(name, artifact_location = NULL, client = NULL) {
  UseMethod("mlflow_create_experiment", client)
}

#' @export
mlflow_create_experiment.mlflow_client <- function(name, artifact_location = NULL, client = NULL) {
  name <- forge::cast_string(name)
  experiment_id <- mlflow_client_create_experiment(client, name, artifact_location)
  invisible(experiment_id)
}

#' @export
mlflow_create_experiment.NULL <- function(name, artifact_location = NULL, client = NULL) {
  mc <- mlflow_client()
  experiment_id <- mlflow_create_experiment.mlflow_client(client, name, artifact_location)
  mlflow_set_active_experiment(experiment_id)
  invisible(experiment_id)
}
