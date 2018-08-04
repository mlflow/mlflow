#' MLflow Command
#'
#' Executes a generic MLflow command through the commmand line interface.
#'
#' @param ... The parameters to pass to the command line.
#' @param background Should this command be triggered as a background task?
#'   Defaults to \code{FALSE}.
#' @param echo Print the standard output and error to the screen? Defaults to
#'   \code{TRUE}, does not apply to background tasks.
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#' mlflow_install()
#'
#' mlflow_cli("server", "--help")
#' }
#'
#' @importFrom processx run
#' @importFrom processx process
#' @importFrom withr with_envvar
#' @export
mlflow_cli <- function(..., background = FALSE, echo = TRUE) {
  args <- list(...)

  verbose <- getOption("mlflow.verbose", FALSE)

  python <- dirname(python_bin())
  mlflow_bin <- file.path(python, "mlflow")

  env <- list(
    PATH = python
  )

  with_envvar(env, {
    if (background) {
      result <- process$new(mlflow_bin, args = unlist(args), echo_cmd = verbose, supervise = TRUE)
    }
    else {
      result <- run(mlflow_bin, args = unlist(args), echo = echo, echo_cmd = verbose)
    }
  })

  invisible(result)
}

mlflow_cli_file_output <- function(response) {
  temp_file <- tempfile(fileext = ".txt")
  writeLines(response$stdout, temp_file)
  temp_file
}
