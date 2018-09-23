#' @export
mlflow_create_experiment.NULL <- function(name, artifact_location = NULL, client = NULL) {
  mc <- mlflow_client()
  experiment_id <- mlflow_create_experiment.mlflow_client(client, name, artifact_location)
  mlflow_set_active_experiment(experiment_id)
  invisible(experiment_id)
}
