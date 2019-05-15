# Returns the current MLflow R package version
mlflow_version <- function() {
  utils::packageVersion("mlflow")
}

# Returns the name of a conda environment in which to install the Python MLflow package
mlflow_conda_env_name <- function() {
  paste("r-mlflow", mlflow_version(), sep = "-")
}

# Create conda env used by MLflow if it doesn't already exist
#' @importFrom reticulate conda_install conda_create conda_list
mlflow_maybe_create_conda_env <- function() {
  conda <- mlflow_conda_bin()
  conda_env_name <- mlflow_conda_env_name()
  if (!conda_env_name %in% conda_list(conda = conda)$name) {
    conda_create(conda_env_name, conda = conda)
  }
}

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
  mlflow_maybe_create_conda_env()
  packages <- python_mlflow_deps()
  conda <- mlflow_conda_bin()
  conda_install(packages, envname = mlflow_conda_env_name(), pip = TRUE, conda = conda)
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
  reticulate::conda_remove(envname = mlflow_conda_env_name(), conda = mlflow_conda_bin())
}


mlflow_conda_bin <- function() {
  conda_home <- Sys.getenv("MLFLOW_CONDA_HOME", NA)
  conda <- if (!is.na(conda_home)) paste(conda_home, "bin", "conda", sep = "/") else "auto"
  conda_binary(conda = conda)
}
