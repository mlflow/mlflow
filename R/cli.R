mlflow_cli_path <- function() {
  result <- pip_run("show", "mlflow", echo = FALSE)
  location_match <- regexec("Location: ([^\n]*)", result$stdout)
  site_path <- regmatches(result$stdout, location_match)[[1]][[2]]
  file.path(site_path, "mlflow", "cli.py")
}

#' MLflow Command
#'
#' Executes a generic MLflow command through the commmand line interface.
#'
#' @param ... The parameters to pass to the command line.
#' @param background Should this command be triggered as a background task?
#'   Defaults to \code{FALSE}.
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
mlflow_cli <- function(..., background = FALSE) {
  args <- list(...)
  args <- c(
    mlflow_cli_path(),
    args
  )

  verbose <- getOption("mlflow.verbose", FALSE)

  python <- python_bin()
  env <- list(
    PATH = dirname(python)
  )

  with_envvar(env, {
    if (background) {
      result <- process$new(python, args = unlist(args), echo_cmd = verbose, supervise = TRUE)
    }
    else {
      result <- run(python, args = unlist(args), echo = TRUE, echo_cmd = verbose)
    }
  })

  invisible(result)
}
