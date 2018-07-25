mlflow_cli_path <- function() {
  result <- pip_run("show", "mlflow", echo = FALSE)
  location_match <- regexec("Location: ([^\n]*)", result$stdout)
  site_path <- regmatches(result$stdout, location_match)[[1]][[2]]
  file.path(site_path, "mlflow", "cli.py")
}

#' @importFrom processx run
#' @importFrom processx process
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

  withr::with_envvar(env, {
    if (background) {
      result <- process$new(python, args = unlist(args), echo_cmd = verbose, supervise = TRUE)
    }
    else {
      result <- run(python, args = unlist(args), echo = verbose, echo_cmd = verbose)
    }
  })

  invisible(result)
}
