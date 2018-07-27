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
#' @importFrom reticulate conda_install conda_create conda_list
#' @export
mlflow_install <- function() {
  packages <- c(
    "pandas",
    "mlflow"
  )

  if (!"r-mlflow" %in% conda_list()$name) conda_create("r-mlflow")
  conda_install(packages, envname = "r-mlflow", pip = TRUE)
}

mlflow_is_installed <- function() {
  python_conda_installed() && "r-mlflow" %in% conda_list()$name
}
