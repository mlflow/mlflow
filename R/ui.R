#' @importFrom utils browseURL
mlflow_view_url <- function(url) {
  getOption("page_viewer", browseURL)(url)
}

#' MLflow User Interface
#'
#' Launches MLflow user interface.
#'
#' @param mc The MLflow connection created using \code{mlflow_connect()}.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' mc <- mlflow_connect()
#' mlflow_ui(mc)
#' }
#'
#' @export
mlflow_ui <- function(mc) {
  mlflow_connection_validate(mc)

  mlflow_view_url(mlflow_url(mc))
}
