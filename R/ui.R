mlflow_view_url <- function(url) {
  getOption("page_viewer", browseURL)(url)
}

mlflow_ui_url <- function() {
  getOption("mlflow.ui", "http://127.0.0.1:5000")
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
#' mlflow_ui()
#' }
#'
#' @export
mlflow_ui <- function() {
  mlflow_cli("ui", background = getOption("mlflow.ui.background", TRUE))

  mlflow_view_url(mlflow_ui_url())
}
