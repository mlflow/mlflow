#' Create Dependency Snapshot
#'
#' Creates a snapshot of all dependencies required to run the files in the
#' current directory.
#'
#' @export
mlflow_snapshot <- function() {
  packrat::.snapshotImpl(
    project = getwd(),
    ignore.stale = getOption("mlflow.snapshot.stale", FALSE),
    prompt = getOption("mlflow.snapshot.prompt", FALSE),
    snapshot.sources = getOption("mlflow.snapshot.sources", FALSE),
    verbose = mlflow_is_verbose(),
    fallback.ok = getOption("mlflow.snapshot.fallback", FALSE)
  )

  file.copy("packrat/packrat.lock", "r-dependencies.txt")
  mlflow_lock_delete()
}

mlflow_lock_delete <- function() {
  if (file.exists("packrat/packrat.lock")) {
    unlink("packrat/packrat.lock")
    if (length(dir("packrat")) == 0) unlink("packrat", recursive = TRUE)
  }
}

#' Restore Snapshot
#'
#' Restores a snapshot of all dependencies required to run the files in the
#' current directory.
#'
#' @export
mlflow_restore_snapshot <- function() {

  if (!file.exists("r-dependencies.txt")) {
    stop("r-dependencies.txt expected but does not exist, run 'mlflow_run()' or 'mlflow_snapshot()'.")
  }

  if (!file.exists("packrat")) dir.create("packrat")
  file.copy("r-dependencies.txt", "packrat/packrat.lock")
  on.exit(mlflow_lock_delete)

  if (nchar(Sys.getenv("MLFLOW_SNAPSHOT_CACHE")) > 0) {
    Sys.setenv(R_PACKRAT_CACHE_DIR = Sys.getenv("MLFLOW_SNAPSHOT_CACHE"))
  }

  options(packrat.verbose.cache = mlflow_is_verbose(), packrat.connect.timeout = 10)

  packrat::set_opts(
    auto.snapshot = FALSE,
    use.cache = TRUE,
    project = getwd(),
    persist = FALSE
  )

  packrat::restore(overwrite.dirty = TRUE,
                   prompt = FALSE,
                   restart = FALSE)

  packrat::on()
}

mlflow_snapshot_warning <- function() {
  warning(
    "Running without restoring the packages snapshot may not reload the model correctly. ",
    "Consider running 'mlflow_restore_snapshot()' or setting the 'restore' parameter to 'TRUE'."
  )
}

mlflow_restore_or_warning <- function(restore) {
  if (restore) {
    mlflow_restore_snapshot()
  } else {
    mlflow_snapshot_warning()
  }
}
