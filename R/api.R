#' Choose REST or CLI API
#'
#' Choose between the REST or CLI APIs and forward parameters
mlflow_choose_api <- function(cli_call, rest_call, ...) {
  if (mlflow_tracking_is_remote())
    rest_call(...)
  else
    cli_call(...)
}
