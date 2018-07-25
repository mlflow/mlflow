mlflow_cli_path <- function() {
  result <- pip_run("show", "mlflow", echo = FALSE)
  location_match <- regexec("Location: ([^\n]*)", r$stdout)
  site_path <- regmatches(r$stdout, location_match)[[1]][[2]]
  file.path(site_path, "mlflow", "cli.py")
}

#' @importFrom processx run
mlflow_cli <- function(...) {
  args <- list(...)
  args <- c(
    mlflow_cli_path(),
    args
  )

  result <- run(python_bin(), args = unlist(args), echo = TRUE)
  invisible(result)
}
