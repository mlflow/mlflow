#' @param client An `mlflow_client` object.
#' @keywords internal
#' @details The Tracking Client family of functions require an MLflow client to be
#'   specified explicitly. These functions allow for greater control of where the
#'   operations take place in terms of services and runs, but are more verbose
#'   compared to the Fluent API.
