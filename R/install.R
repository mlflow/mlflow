#' Install MLflow
#'
#' Installs MLflow for individual use.
#'
#' Notice that MLflow requires Python and Conda to be installed,
#' see \url{https://www.python.org/getit/} and \url{https://conda.io/docs/installation.html}.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#' }
#'
#' @importFrom reticulate conda_install
#' @importFrom reticulate py_install
#' @export
mlflow_install <- function() {
  packages <- c(
    "pandas",
    "mlflow"
  )

  conda_install(packages, envname = "r-mlflow", pip = TRUE)
}
