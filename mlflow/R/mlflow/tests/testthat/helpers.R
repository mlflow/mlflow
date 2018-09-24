mlflow_clear_test_dir <- function(path) {
  if (dir.exists(path)) {
    unlink(path, recursive = TRUE)
  }
  mlflow_disconnect()
}
