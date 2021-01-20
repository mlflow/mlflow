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

as_uri <- function(file_path) {
  if (.Platform$OS.type == "windows") {
    # normalize file path such as c:/directory into file://c:/directory on
    # Windows so that MLflow CLI does not confuse 'c' as the scheme of the URI
    # Also, replacing back slashes with forward slashes
    gsub("\\\\", "/", paste0("file://", normalizePath(file_path)))
  } else {
    # UNIX file paths such as ., dir1, dir1/dir2, / or, /dir1/dir2 etc do not
    # have this problem so we don't need to do anything
    file_path
  }
}

skip_on_windows <- function(reason = NULL) {
  if (identical(.Platform$OS.type, "windows")) {
    msg <- "Test will be skipped on Windows."
    if (!is.null(reason)) {
      msg <- paste(msg, "Reason:", reason)
    }
    skip(msg)
  }
}
