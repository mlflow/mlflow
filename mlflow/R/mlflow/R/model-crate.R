#' @rdname mlflow_save_model
#' @export
mlflow_save_model.crate <- function(model, path, model_spec=list(), ...) {
  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  serialized <- serialize(model, NULL)

  saveRDS(
    serialized,
    file.path(path, "crate.bin")
  )

  model_spec$flavors <- append(model_spec$flavors, list(
    crate = list(
      version = "0.1.0",
      model = "crate.bin"
    )
  ))
  mlflow_write_model_spec(path, model_spec)
  model_spec
}

#' @export
mlflow_load_flavor.mlflow_flavor_crate <- function(flavor, model_path) {
  unserialize(readRDS(file.path(model_path, "crate.bin")))
}

#' @export
mlflow_predict.crate <- function(model, data, ...) {
  do.call(model, list(data, ...))
}
