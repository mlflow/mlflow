# new_mlflow_rest_data <- function(data, ..., class = character()) {
#   structure(
#     data,
#     ...,
#     class = c(class, "mlflow_rest_data")
#   )
# }
#
# new_mlflow_rest_data_experiment <- function(data) {
#   new_mlflow_rest_data(tibble::as_tibble(data), class = "mlflow_rest_data_experiment")
# }

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

new_mlflow_rest_data_array <- function(data, type = NULL, class = character()) {
  type <- type %||% class(data[[1]])[[1]]
  structure(
    data,
    type = type,
    class = c(class, "mlflow_rest_data_array")
  )
}

#' @export
print.mlflow_rest_data_array <- function(x, ...) {
  print_type(x)
  out <- x %>%
    purrr::transpose() %>%
    purrr::map(unlist) %>%
    purrr::map_at("timestamp", milliseconds_to_date) %>%
    tibble::as_tibble()
  print(out)
  invisible(x)
}

new_mlflow_rest_data_array_metric <- function(data) {
  new_mlflow_rest_data_array(
    data,
    type = "Metric",
    class = "mlflow_rest_data_array_metric"
  )
}

print_type <- function(x) {
  print(
    glue::glue("
             MLflow array of type `{type}`

               ",
      type = attr(x, "type")
    )
  )
}

#' #' @export
#' print.mlflow_rest_data_array_metric <- function(x, ...) {
#'   print_type(x)
#'   out <- x %>%
#'     purrr::transpose() %>%
#'     purrr::map(unlist) %>%
#'     purrr::map_at("timestamp", milliseconds_to_date) %>%
#'     tibble::as_tibble()
#'   print(out)
#'   invisible(x)
#' }
