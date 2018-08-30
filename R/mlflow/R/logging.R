mlflow_is_verbose <- function() {
  nchar(Sys.getenv("MLFLOW_VERBOSE")) > 0 || getOption("mlflow.verbose", FALSE)
}

mlflow_verbose_message <- function(...) {
  if (mlflow_is_verbose()) {
    message(...)
  }
}
