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
mlflow_save_model <- function(f, path = "model") {
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

#' Predict using MLflow Model
#'
#' Predict using a MLflow Model from a 'JSON' file.
#'
#' @param model_dir The path to the MLflow model, as a string.
#' @param data_file 'JSON' file containing data frame to be used for prediction.
#' @param restore Should \code{mlflow_restore()} be called before serving?
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#'
#' # save simple model which roundtrips data as prediction
#' mlflow_save_model(function(df) df, "mlflow_roundtrip")
#'
#' # save data as json
#' jsonlite::write_json(iris, "iris.json")
#'
#' # serve an existing model over a web interface
#' mlflow_predict("mlflow_roundtrip", "iris.json")
#' }
#'
#' @export
mlflow_predict <- function(
  model_dir,
  data_file,
  restore = FALSE
) {
  if (restore) mlflow_restore()

  model <- mlflow_load_model(model_dir)
  mlflow_predict_model()
}
