#' Install MLflow
#'
#' Installs MLflow for individual use.
#'
#' Notice that MLflow requires Python to be installed,
#' see \url{https://www.python.org/getit/}.
#'
#' @importFrom reticulate conda_install
#' @export
mlflow_install <- function() {
  packages <- c(
    "pandas",
    "mlflow"
  )

  if (python_use_conda()) {
    conda_install(packages, envname = "r-mlflow", pip = TRUE)
  }
  else {
    py_install(packages, envname = "r-mlflow")
  }
}
