#' @export
mlflow_save_model.crate <- function(x, path = "model") {
  serialized <- serialize(x, NULL)

  saveRDS(
    serialized,
    file.path(path, "r_model.bin")
  )

  mlflow_write_model_spec(
    path,
    list(
      flavors = list(
        r_function = list(
          version = "0.1.0",
          model = "r_model.bin"
        )
      )
    )
  )
}
