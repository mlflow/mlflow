#' @importFrom reticulate conda_list
python_bin_conda <- function() {
  envs <- conda_list()
  mlflow_env <- envs[envs$name == "r-mlflow",]
  if (nrow(mlflow_env) == 0) {
    stop("MLflow not configured, please run mlflow_install().")
  }

  mlflow_env$python
}

python_bin <- function(conda = python_use_conda()) {
  python <- python_bin_conda()
  path.expand(python)
}

#' @importFrom processx run
pip_run <- function(..., echo = TRUE) {
  args <- list(...)

  command <- file.path(dirname(python_bin()), "pip")
  result <- run(command, args = unlist(args), echo = echo)

  invisible(result)
}
