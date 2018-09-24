mlflow_clear_test_dir <- function(path) {
  if (dir.exists(path)) {
    unlink(path, recursive = TRUE)
  }
  mlflow_deregister_local_servers()
}

mlflow_deregister_local_servers <- function() {
  purrr::walk(as.list(.globals$url_mapping), ~ .x$handle$kill())
  .globals$url_mapping <- NULL
}
