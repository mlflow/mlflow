#' Read Command Line Parameter
#'
#' Reads a command line parameter.
#'
#' @param name The name for this parameter.
#' @param default The default value for this parameter.
#' @param type Type of this parameter. Required if `default` is not set.
#' @param description Optional description for this parameter.
#'
#' @export
mlflow_param <- function(name, default = NULL, type = NULL, description = NULL) {
  if (is.null(default) && is.null(type))
    stop("At least one of `default` or `type` must be specified", call. = FALSE)
  if (!is.null(default) && !is.null(type) && !identical(typeof(default), type))
    stop("`default` value is not of type ", type, ".", call. = FALSE)
  .globals$run_params[[name]] %||% default
}
