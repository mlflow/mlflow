mlflow_cli_path <- function() {
  result <- pip_run("show", "mlflow", echo = FALSE)
  location_match <- regexec("Location: ([^\n]*)", r$stdout)
  site_path <- regmatches(r$stdout, location_match)[[1]][[2]]
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

  if (background) {
    result <- process$new(python_bin(), args = unlist(args), echo = FALSE, supervise = TRUE)
  }
  else {
    result <- run(python_bin(), args = unlist(args), echo = TRUE)
  }

  invisible(result)
}
