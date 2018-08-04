#' Dependencies Snapshot
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
    verbose = getOption("mlflow.verbose", FALSE),
    fallback.ok = getOption("mlflow.snapshot.fallback", FALSE)
  )

  if (file.exists("packrat/packrat.lock")) {
    file.copy("packrat/packrat.lock", "r-dependencies.txt")
    unlink("packrat/packrat.lock")
    if (length(dir("packrat")) == 0) unlink("packrat", recursive = TRUE)
  }
}
