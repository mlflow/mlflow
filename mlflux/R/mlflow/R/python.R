# Computes path to Python executable within conda environment created for the MLflow R package
#' @importFrom reticulate conda_list
get_python_bin <- function() {
  in_env <- Sys.getenv("MLFLOW_PYTHON_BIN")
  if (in_env != "") {
    return(in_env)
  }
  conda <- mlflow_conda_bin()
  envs <- conda_list(conda = conda)
  mlflow_env <- envs[envs$name == mlflow_conda_env_name(), ]
  if (nrow(mlflow_env) == 0) {
    stop(paste("MLflow not configured, please run install_mlflow() or ",
               "set MLFLOW_PYTHON_BIN and MLFLOW_BIN environment variables.", sep = ""))
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

# Returns path to MLflow CLI, assumed to be in the same bin/ directory as the
# Python executable
python_mlflow_bin <- function() {
  in_env <- Sys.getenv("MLFLOW_BIN")
  if (in_env != "") {
    return(in_env)
  }
  python_bin_dir <- dirname(python_bin())
  file.path(python_bin_dir, "mlflow")
}

# Return path to conda home directory, such that the `conda` executable can be found
# under conda_home/bin/
#' @importFrom reticulate conda_binary
python_conda_home <- function() {
  path <- try(dirname(dirname(mlflow_conda_bin())), silent = TRUE)
  if (class(path) == "try-error") {
    return(NA)
  }
  path
}
