#' @export
mlflow_save_flavor.crate <- function(x, path = "model", r_dependencies=NULL, conda_env=NULL) {
  serialized <- serialize(x, NULL)

  saveRDS(
    serialized,
    file.path(path, "crate.bin")
  )

  res <- list(
    crate = list(
      version = "0.1.0",
      model = "crate.bin"
    )
  )
  if (!is.null(r_dependencies)) {
    dep_file <- basename(r_dependencies)
    if (dep_file != "r-dependencies.txt") {
      stop("Dependency", dep_file,
           "is unsupported by cran flavor. R-dependencies must be named 'r-dependencies.txt'")
    }
    dst <- file.path(path, basename(conda_env))
    if (r_dependencies != dst) {
      file.copy(from = r_dependencies, to = dst)
      res$crate$r_dependencies <- basename(r_dependencies)
    }
  }
  if (!is.null(conda_env)){
    dst <- file.path(path, basename(conda_env))
    if (conda_env != dst) {
      file.copy(from = conda_env, to = dst)
      res$crate$conda_env <- basename(conda_env)
    }
  }
  res
}

#' @export
mlflow_load_flavor.crate <- function(model_path) {
  unserialize(readRDS(file.path(model_path, "crate.bin")))
}

#' @export
mlflow_predict_flavor.crate <- function(model, data) {
  model(data)
}
