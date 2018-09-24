mlflow_clear_test_dir <- function(path) {
  mlflow_end_run()
  mlflow_set_active_experiment(NULL)
  if (dir.exists(path)) {
    unlink(path, recursive = TRUE)
  }
  mlflow_deregister_local_servers()
}

mlflow_deregister_local_servers <- function() {
  purrr::walk(as.list(mlflow:::.globals$url_mapping), ~ .x$handle$kill())
  rlang::env_unbind(mlflow:::.globals, "url_mapping")
}
