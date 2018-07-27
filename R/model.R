#' @importFrom yaml write_yaml
mlflow_save_model <- function(f, path = "mlflow-model") {
  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  model_raw <- serialize(
    list(
      r_function = f,
      r_environment = as.list(.GlobalEnv)
    ),
    NULL
  )

  saveRDS(
    model_raw,
    file.path(path, "r_model.bin")
  )

  write_yaml(
    list(
      time_created = Sys.time(),
      flavors = list(
        r_function = list(
          version = "0.1.0",
          model = "r_model.bin"
        )
      )
    ),
    file.path(path, "MLmodel")
  )
}
