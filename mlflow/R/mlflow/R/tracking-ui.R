#' @importFrom utils browseURL
mlflow_view_url <- function(url) {
  getOption("page_viewer", browseURL)(url)

  invisible(url)
}

#' Run MLflow User Interface
#'
#' Launches the MLflow user interface.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#'
#' # launch mlflow ui locally
#' mlflow_ui()
#'
#' # launch mlflow ui for existing mlflow server
#' mlflow_set_tracking_uri("http://tracking-server:5000")
#' mlflow_ui()
#' }
#'
#' @template roxlate-client
#' @param ... Optional arguments passed to `mlflow_server()` when `x` is a path to a file store.
#' @export
mlflow_ui <- function(client, ...) {
  UseMethod("mlflow_ui")
}

#' @export
mlflow_ui.mlflow_client <- function(client, ...) {
  mlflow_view_url(client$get_host_creds()$host)
}

#' @export
mlflow_ui.NULL <- function(client, ...) {
  client <- mlflow_client()
  mlflow_ui(client)
}
