mlflow_clear_test_dir <- function(path_or_uri = NULL) {
  purrr::safely(mlflow_end_run)()
  mlflow:::mlflow_set_active_experiment_id(NULL)
  
  if (!is.null(path_or_uri)) {
    # Handle SQLite URI cleanup
    if (grepl("^sqlite:///", path_or_uri)) {
      db_file <- sub("^sqlite:///", "", path_or_uri)
      if (file.exists(db_file)) {
        file.remove(db_file)
      }
    } else if (dir.exists(path_or_uri)) {
      # Legacy file store cleanup (can be removed later)
      unlink(path_or_uri, recursive = TRUE)
    }
  }
  
  deregister_local_servers()
}

deregister_local_servers <- function() {
  purrr::walk(as.list(mlflow:::.globals$url_mapping), ~ .x$handle$kill())
  rlang::env_unbind(mlflow:::.globals, "url_mapping")
}
