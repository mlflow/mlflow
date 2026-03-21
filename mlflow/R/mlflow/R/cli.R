# Runs a generic MLflow command through the command-line interface.
#
# @param ... The parameters to pass to the command line.
# @param background Should this command be triggered as a background task?
#   Defaults to \code{FALSE}.
# @param echo Print the standard output and error to the screen? Defaults to
#   \code{TRUE}, does not apply to background tasks.
# @param stderr_callback \code{NULL} (the default), or a function to call for 
#   every chunk of the standard error, passed to \code{\link[=processx:run]{processx::run()}}.
# @param client MLflow client to provide environment for the cli process.
#
# @return A \code{processx} task.
#' @importFrom processx run
#' @importFrom processx process
#' @importFrom withr with_envvar
mlflow_cli <- function(...,
                       background = FALSE,
                       echo = TRUE,
                       stderr_callback = NULL,
                       client = mlflow_client()) {
  env <- if (is.null(client)) list() else client$get_cli_env()
  args <- list(...)
  verbose <- mlflow_is_verbose()
  python <- dirname(python_bin())
  mlflow_bin <- python_mlflow_bin()
  env <- modifyList(list(
    PATH = paste(python, Sys.getenv("PATH"), sep = ":"),
    MLFLOW_TRACKING_URI = mlflow_get_tracking_uri(),
    MLFLOW_BIN = mlflow_bin,
    MLFLOW_PYTHON_BIN = python_bin()
  ), env)
  MLFLOW_CONDA_HOME <- Sys.getenv("MLFLOW_CONDA_HOME", NA)
  if (!is.na(MLFLOW_CONDA_HOME)) {
    env$MLFLOW_CONDA_HOME <- MLFLOW_CONDA_HOME
  }
  with_envvar(env, {
    if (background) {
      result <- process$new(mlflow_bin, args = unlist(args), echo_cmd = verbose, supervise = TRUE)
    } else {
      result <- run(mlflow_bin, args = unlist(args), echo = echo, echo_cmd = verbose, stderr_callback = stderr_callback)
    }
  })
  invisible(result)
}

mlflow_cli_file_output <- function(response) {
  temp_file <- tempfile(fileext = ".txt")
  writeLines(response$stdout, temp_file)
  temp_file
}
