mlflow_cli_path <- function() {
  result <- python_run(c("pip3", "pip"), "show", "mlflow", echo = FALSE)
  location_match <- regexec("Location: ([^\n]*)", r$stdout)
  site_path <- regmatches(r$stdout, location_match)[[1]][[2]]
  file.path(site_path, "mlflow", "cli.py")
}

mlflow_cli <- function(command) {
  python_run(c("python3", "python"), mlflow_cli_path(), command)
}
