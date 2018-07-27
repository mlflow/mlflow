mlflow_run <- function(x) {
  UseMethod("mlflow_run")
}

mlflow_run.character <- function(x) {
  "file"
}

mlflow_run.function <- function(x) {
  "function"
}
