#' @importFrom reticulate conda_list
python_bin_conda <- function() {
  conda = Sys.getenv("MLFLOW_CONDA_HOME", "auto")
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
    conda_binary(conda = Sys.getenv("MLFLOW_CONDA_HOME", "auto"))
    TRUE
  }, error = function(err) {
    FALSE
  })
}

#' @importFrom reticulate conda_binary
python_conda_bin <- function() {
  dirname(conda_binary(conda = Sys.getenv("MLFLOW_CONDA_HOME", "auto")))
}

python_conda_home <- function() {
  dirname(python_conda_bin())
}
