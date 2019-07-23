# Gets path to Python provided as an argument when running the script
get_bin_from_args <- function(bin) {
  in_args <- grepl(paste0("--", bin), commandArgs())
  if (any(in_args)) {
    path <- commandArgs()[in_args]
    path <- trimws(strsplit(path, "=")[[1]][2])
  } else {
    path <- NULL
  }
  path
}

# Computes path to Python executable within conda environment created for the MLflow R package
#' @importFrom reticulate conda_list
get_python_bin <- function() {
  if ("--no-conda" %in% commandArgs()) {
    path <- get_bin_from_args("python")
    if (is.null(path)) {
      stop(paste("When using --no-conda option you also need to provide path to",
                 "Python and MLflow using --python=<path> and --mlflow=<path>"))
    }
  } else {
    conda <- mlflow_conda_bin()
    envs <- conda_list(conda = conda)
    mlflow_env <- envs[envs$name == mlflow_conda_env_name(), ]
    if (nrow(mlflow_env) == 0) {
      stop("MLflow not configured, please run install_mlflow().")
    }
    path <- mlflow_env$python
  }
  path
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
  path <- get_bin_from_args("mlflow")
  if (is.null(path) && ("--no-conda" %in% commandArgs())) {
    stop(paste("When using --no-conda option you also need to provide path to",
               "Python and MLflow using --python=<path> and --mlflow=<path>"))
  } else if (is.null(path)) {
    python_bin_dir <- dirname(python_bin())
    path <- file.path(python_bin_dir, "mlflow")
  }
  path
}

# Return path to conda home directory, such that the `conda` executable can be found
# under conda_home/bin/
#' @importFrom reticulate conda_binary
python_conda_home <- function() {
  if ("--no-conda" %in% commandArgs()) {
    conda_home <- NULL
  } else {
    conda_home <- dirname(dirname(mlflow_conda_bin()))
  }
  conda_home
}
