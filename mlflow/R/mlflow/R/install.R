#' Install MLflow
#'
#' Installs MLflow for individual use.
#'
#' MLflow requires Python and Conda to be installed.
#' See \url{https://www.python.org/getit/} and \url{https://docs.conda.io/projects/conda/en/latest/user-guide/install/}.
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
  conda <- mlflow_conda_bin()
  if (!"r-mlflow" %in% conda_list(conda = conda)$name) {
    conda_create("r-mlflow", conda = conda)
    conda_install(packages, envname = "r-mlflow", pip = TRUE, conda = conda)
  }
}

#' Uninstall MLflow
#'
#' Uninstalls MLflow by removing the Conda environment.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#' mlflow_uninstall()
#' }
#'
#' @importFrom reticulate conda_install conda_create conda_list
#' @export
mlflow_uninstall <- function() {
  reticulate::conda_remove(envname = "r-mlflow", conda = mlflow_conda_bin())
}


mlflow_conda_bin <- function() {
  conda_home <- Sys.getenv("MLFLOW_CONDA_HOME", NA)
  conda <- if (!is.na(conda_home)) paste(conda_home, "bin", "conda", sep = "/") else "auto"
  conda_binary(conda = conda)
}
