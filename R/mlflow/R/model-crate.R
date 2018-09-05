#' @export
mlflow_save_flavor.crate <- function(x, path = "model") {
  serialized <- serialize(x, NULL)

  saveRDS(
    serialized,
    file.path(path, "r_crate.bin")
  )

  list(
    r_crate = list(
      version = "0.1.0",
      model = "r_crate.bin"
    )
  )
}

#' @export
mlflow_load_flavor.crate <- function(model_path) {
  unserialize(readRDS(model_path))
}

#' @export
mlflow_predict_flavor.crate <- function(model, data) {
  model(data)
}
