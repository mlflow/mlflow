#' Save Model for MLflow
#'
#' Saves model in MLflow's format that can later be used
#' for prediction and serving.
#'
#' @param f The serving function that will perform a prediction.
#' @param path Destination path where this MLflow compatible model
#'   will be saved.
#'
#' @importFrom yaml write_yaml
#' @export
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

#' @export
mlflow_load_model <- function(model_dir) {
  list(
    r_function = function(df) 1,
    r_environment = list()
  )
}

#' @export
mlflow_predict_model <- function(model, df) {
  model$r_function(df)
}
