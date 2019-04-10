new_mlflow_rest_data <- function(data, ..., class = character()) {
  structure(
    data,
    ...,
    class = c(class, "mlflow_rest_data")
  )
}

new_mlflow_rest_data_experiment <- function(data) {
  new_mlflow_rest_data(data, class = "mlflow_rest_data_experiment")
}

#' @export
print.mlflow_rest_data_experiment <- function(x, ...) {
  print(glue::glue("
           MLflow experiment \"{name}\" (id: {id})
             Artifact location: {artifact_location}
             Lifecycle stage: {lifecycle_stage}
           ",
    name = x$name,
    id = x$experiment_id,
    artifact_location = x$artifact_location,
    lifecycle_stage = x$lifecycle_stage
  ))
  invisible(x)
}
