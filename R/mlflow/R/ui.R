#' @importFrom utils browseURL
mlflow_view_url <- function(url) {
  getOption("page_viewer", browseURL)(url)
}

#' MLflow User Interface
#'
#' Launches MLflow user interface.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' # launch mlflow ui locally
#' mlflow_ui()
#'
#' # launch mlflow ui for existing mlflow server
#' mlflow_set_tracking_uri("http://tracking-server:5000")
#' mlflow_ui()
#' }
#'
#' @param x If specified, can be either an `mlflow_connection` object or a string
#'   specifying the file store, i.e. the root of the backing file store for experiment
#'   and run data.
#' @param ... Optional arguments passed to `mlflow_server()` when `x` is a path to a file store.
#' @export
mlflow_ui <- function(x, ...) {
  UseMethod("mlflow_ui")
}

#' @export
mlflow_ui.character <- function(x, ...) {
  file_store <- fs::path_abs(x)
  active_mc <- mlflow_active_connection()
  url <- if (!is.null(active_mc) && identical(active_mc$file_store, file_store)) {
    active_mc$url
  } else {
    mc <- mlflow_connect(file_store = file_store, ...)
    mc$url
  }

  mlflow_view_url(url)
}

#' @export
mlflow_ui.mlflow_connection <- function(x, ...) {
  mlflow_view_url(x$url)
}

#' @export
mlflow_ui.NULL <- function(x, ...) {
  mlflow_view_url(mlflow_tracking_url_get())
}
