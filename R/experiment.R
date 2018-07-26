#' List Experiments
#'
#' Retrieves MLflow experiments as a data frame.
#'
#' @param mc The MLflow connection created using \code{mlflow_connect()}.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' mc <- mlflow_connect()
#' mlflow_experiments(mc)
#' }
#'
#' @export
mlflow_experiments <- function(mc) {
  response <- mlflow_api(mc, "experiments", "list")
  response$experiments
}
