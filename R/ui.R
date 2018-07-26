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
#' mlflow_tracking_url("http://tracking-server:5000")
#' mlflow_ui()
#' }
#'
#' @export
mlflow_ui <- function() {
  mc <- mlflow_connection_active_get()
  if (is.null(mc)) {
    mc <- mlflow_connect()
    mlflow_connection_active_set(mc)
  }

  mlflow_view_url(mlflow_connection_url(mc))
}
