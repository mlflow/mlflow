#' @param client An `mlflow_client` object. See details.
#' @param ... Additional arguments; currently not used.
#' @details For functions where `client` is optional, providing it will invoke the
#'   Tracking Service API, allowing the user to specify the service where
#'   the operation is executed. When `client` isn't specified, it defaults to
#'   the service set by `mlflow_set_tracking_uri()`.
