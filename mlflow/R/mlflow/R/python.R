#' @importFrom reticulate conda_list
python_bin_conda <- function() {
  conda <- mlflow_conda_bin()
  envs <- conda_list(conda = conda)
  mlflow_env <- envs[envs$name == "r-mlflow", ]
  if (nrow(mlflow_env) == 0) {
    stop("MLflow not configured, please run mlflow_install().")
  }
  mlflow_env$python
}

python_bin <- function() {
  if (is.null(.globals$python_bin)) {
    python <- python_bin_conda()
    .globals$python_bin <- path.expand(python)
  }

  .globals$python_bin
}

#' @importFrom processx run
pip_run <- function(..., echo = TRUE) {
  args <- list(...)

  command <- file.path(dirname(python_bin()), "pip")
  result <- run(command, args = unlist(args), echo = echo)

  invisible(result)
}

#' @importFrom reticulate conda_binary
python_conda_installed <- function() {
  tryCatch({
    mlflow_conda_bin()
    TRUE
  }, error = function(err) {
    FALSE
  })
}

#' @importFrom reticulate conda_binary
python_conda_bin <- function() {
  dirname(mlflow_conda_bin())
}

python_conda_home <- function() {
  dirname(python_conda_bin())
}
