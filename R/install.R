#' Install MLflow
#'
#' Installs MLflow for individual use.
#'
#' Notice that MLflow requires Python and pip to be installed.
#' See \url{https://www.python.org/getit/} and \url{https://pip.pypa.io/}.
#'
#'  @export
mlflow_install <- function() {
  python_run(c("pip3", "pip"), "install", "--user", "pandas")
  python_run(c("pip3", "pip"), "install", "--user", "mlflow")
}
