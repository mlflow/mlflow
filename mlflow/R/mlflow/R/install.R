# Returns the current MLflow R package version
mlflow_version <- function() {
  utils::packageVersion("mlflow")
}

# Returns the name of a conda environment in which to install the Python MLflow package
mlflow_conda_env_name <- function() {
  paste("r-mlflow", mlflow_version(), sep = "-")
}

#' Create conda env used by MLflow if it doesn't already exist
#'
#' @importFrom reticulate conda_install conda_create conda_list
#' @param python_version Python version to use within conda environment created for
#' installing the MLflow CLI.
mlflow_maybe_create_conda_env <- function(python_version) {
  packages <- c(paste("python", python_version, sep = "="))
  conda <- mlflow_conda_bin()
  conda_env_name <- mlflow_conda_env_name()
  if (!conda_env_name %in% conda_list(conda = conda)$name) {
    message("Creating conda environment ", conda_env_name)
    conda_create(conda_env_name, conda = conda, packages = packages)
  }
}

#' Install MLflow
#'
#' Installs auxiliary dependencies of MLflow (e.g. the MLflow CLI). As a
#' one-time setup step, you must run install_mlflow() to install these
#' dependencies before calling other MLflow APIs.
#'
#' install_mlflow() requires Python and Conda to be installed.
#' See \url{https://www.python.org/getit/} and \url{https://docs.conda.io/projects/conda/en/latest/user-guide/install/}.
#'
#' Alternatively, you can set MLFLOW_PYTHON_BIN and MLFLOW_BIN environment variables
#' instead. MLFLOW_PYTHON_BIN should point to python executable and MLFLOW_BIN to mlflow cli
#' executable. These variables allow you to use custom mlflow installation. Note that there may be
#' some compatibility issues if the custom mlflow version does not match the version of the R
#' package.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' install_mlflow()
#' }
#'
#'
#' @importFrom reticulate conda_install conda_create conda_list
#' @param python_version Optional Python version to use within conda environment created for
#' installing the MLflow CLI. If unspecified, defaults to using Python 3.6
#' @export
install_mlflow <- function(python_version = "3.6") {
  mlflow_maybe_create_conda_env(python_version)
  # Install the Python MLflow package with version == the current R package version
  packages <- c(paste("mlflow", "==", mlflow_version(), sep = ""))
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
#' install_mlflow()
#' uninstall_mlflow()
#' }
#'
#' @importFrom reticulate conda_install conda_create conda_list
#' @export
uninstall_mlflow <- function() {
  reticulate::conda_remove(envname = mlflow_conda_env_name(), conda = mlflow_conda_bin())
}


mlflow_conda_bin <- function() {
  conda_home <- Sys.getenv("MLFLOW_CONDA_HOME", NA)
  conda <- if (!is.na(conda_home)) paste(conda_home, "bin", "conda", sep = "/") else "auto"
  conda_try <- try(conda_binary(conda = conda), silent = TRUE)
  if (class(conda_try) == "try-error") {
    msg <- paste(attributes(conda_try)$condition$message,
                 paste("  If you are not using conda, you can set the environment variable",
                 "MLFLOW_PYTHON_BIN to the path of your python executable."),
                 sep = "\n")
    stop(msg)
  }
  conda_try
}
