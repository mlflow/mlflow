# Computes path to Python executable within conda environment created for the MLflow R package
#' @importFrom reticulate conda_list
get_python_bin <- function() {
  in_env <- Sys.getenv("MLFLOW_PYTHON_BIN")
  if (file.exists(in_env)) {
    return(in_env)
  }
  conda <- mlflow_conda_bin()
  envs <- conda_list(conda = conda)
  mlflow_env <- envs[envs$name == mlflow_conda_env_name(), ]
  if (nrow(mlflow_env) == 0) {
    stop("MLflow not configured, please run install_mlflow().")
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
  if (file.exists(in_env)) {
    return(in_env)
  }
  python_bin_dir <- dirname(python_bin())
  path <- file.path(python_bin_dir, "mlflow")
  if (!file.exists(path) && file.exists(Sys.getenv("MLFLOW_PYTHON_BIN"))) {
    stop("Mlflow executable does not seem to be in the same directory than Python.\n",
         "  Please set the environment variable MLFLOW_BIN to the path of your mlflow executable."
         )
  }
  path
}

# Return path to conda home directory, such that the `conda` executable can be found
# under conda_home/bin/
#' @importFrom reticulate conda_binary
python_conda_home <- function() {
  if(file.exists(python_bin()) && file.exists(python_mlflow_bin())) {
    return("")
  }
  dirname(dirname(mlflow_conda_bin()))
}
