mlflow_clear_test_dir <- function(path) {
  purrr::safely(mlflow_end_run)()
  mlflow:::mlflow_set_active_experiment_id(NULL)
  if (dir.exists(path)) {
    unlink(path, recursive = TRUE)
  }
  deregister_local_servers()
}

deregister_local_servers <- function() {
  purrr::walk(as.list(mlflow:::.globals$url_mapping), ~ .x$handle$kill())
  rlang::env_unbind(mlflow:::.globals, "url_mapping")
}
