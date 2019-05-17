#' @export
mlflow_save_flavor.crate <- function(model, path = "model", conda_env=NULL) {
  serialized <- serialize(model, NULL)

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
