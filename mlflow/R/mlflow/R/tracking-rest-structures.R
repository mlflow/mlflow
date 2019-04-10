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

new_mlflow_rest_data_array <- function(data, type = NULL) {
  type <- type %||% class(data[[1]])[[1]]
  structure(
    data,
    type = type,
    class = "mlflow_rest_data_array"
  )
}

#' @export
print.mlflow_rest_data_array <- function(x, ...) {
  print(
    glue::glue("
             MLflow array of type `{type}`

               ",
      type = attr(x, "type")
    )
  )

  print(purrr::map_df(x, ~ .x %>% unclass() %>% tibble::as_tibble()))
  invisible(x)
}
