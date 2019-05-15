# Computes path to Python executable within conda environment created for the MLflow R package
#' @importFrom reticulate conda_list
get_python_bin <- function() {
  conda <- mlflow_conda_bin()
  envs <- conda_list(conda = conda)
  mlflow_env <- envs[envs$name == mlflow_conda_env_name(), ]
  if (nrow(mlflow_env) == 0) {
    stop("MLflow not configured, please run mlflow_install().")
  }
  mlflow_env$python
}

# Returns path to Python executable within conda environment created for the MLflow R package
python_bin <- function() {
  if (is.null(.globals$python_bin)) {
    python <- get_python_bin()
    .globals$python_bin <- path.expand(python)
  }

  .globals$python_bin
}

# Return python package dependencies of the MLflow R API.
# @param mlflow_package If provided, pip-installable string referencing the MLflow python package.
#   Defaults to 'mlflow==<version>', where <version> is the current R package version
python_mlflow_deps <- function(mlflow_package = NULL) {
  if (is.null(mlflow_package)) {
    # By default install the Python MLflow package with version == the current R package version
    mlflow_package <- paste("mlflow", "==", mlflow_version(), sep = "")
  }
  c(
    "pandas",
    mlflow_package
  )
}

# Returns path to MLflow CLI, assumed to be in the same bin/ directory as the
# Python executable
python_mlflow_bin <- function() {
  python_bin_dir <- dirname(python_bin())
  file.path(python_bin_dir, "mlflow")
}

# Return path to conda home directory, such that the `conda` executable can be found
# under conda_home/bin/
#' @importFrom reticulate conda_binary
python_conda_home <- function() {
  dirname(dirname(mlflow_conda_bin()))
}
