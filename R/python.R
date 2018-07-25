#' @importFrom reticulate conda_list
python_bin_conda <- function() {
  envs <- conda_list()
  mlflow_env <- envs[envs$name == "r-mlflow",]
  if (nrow(mlflow_env) == 0) {
    stop("MLflow not configured, please run mlflow_install().")
  }

  mlflow_env$python
}

#' @importFrom reticulate get_virtualenv
python_bin_virtualenv <- function() {
  python <- get_virtualenv("r-mlflow")

  if (!file.exists(python)) {
    stop("MLflow not configured, please run mlflow_install().")
  }

  python
}

#' @importFrom reticulate find_conda
python_use_conda <- function() {
  getOption("mlflow.conda", !is.null(find_conda()))
}

python_bin <- function(conda = python_use_conda()) {
  python <- ifelse(conda, python_bin_conda(), python_bin_virtualenv())
  path.expand(python)
}

#' @importFrom processx run
pip_run <- function(..., echo = TRUE) {
  args <- list(...)

  command <- file.path(dirname(python_bin()), "pip")
  result <- run(command, args = unlist(args), echo = echo)

  invisible(result)
}
