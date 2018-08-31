#' Save Model for MLflow
#'
#' Saves model in MLflow's format that can later be used
#' for prediction and serving.
#'
#' @param fn The serving function that will perform a prediction.
#' @param path Destination path where this MLflow compatible model
#'   will be saved.
#'
#' @importFrom yaml write_yaml
#' @export
mlflow_save_model <- function(fn, path = "model") {

  if (!inherits(fn, "crate")) {
    stop("Serving function must be crated using mlflow::crate().")
  }

  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  serialized <- serialize(fn, NULL)

  saveRDS(
    serialized,
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

mlflow_load_model <- function(model_path) {
  spec <- yaml::read_yaml(fs::path(model_path, "MLmodel"))

  if (!"r_function" %in% names(spec$flavors))
    stop("Model must define r_function to be used from R.")

  unserialize(readRDS(fs::path(model_path, spec$flavors$r_function$model)))
}

mlflow_rfunc_predict_impl <- function(model, data) {
  if (!is.data.frame(data))
    stop("Could not load data as a data frame.")

  if (!inherits(model, "crate")) {
    stop("MLflow rfunc model expected to be crated using mlflow::crate().")
  }

  model(data)
}

#' Predict using RFunc MLflow Model
#'
#' Predict using an RFunc MLflow Model from a file or data frame.
#'
#' @param model_path The path to the MLflow model, as a string.
#' @param run_id Run ID of run to grab the model from.
#' @param input_path Path to 'JSON' or 'CSV' file to be used for prediction.
#' @param output_path 'JSON' or 'CSV' file where the prediction will be written to.
#' @param data Data frame to be scored. This can be utilized for testing purposes and can only
#'   be specified when `input_path` is not specified.
#' @param restore Should \code{mlflow_restore_snapshot()} be called before serving?
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
#' # predict existing model from json data
#' mlflow_rfunc_predict("mlflow_roundtrip", "iris.json")
#' }
#'
#' @importFrom utils read.csv
#' @importFrom utils write.csv
#' @export
mlflow_rfunc_predict <- function(
  model_path,
  run_id = NULL,
  input_path = NULL,
  output_path = NULL,
  data = NULL,
  restore = FALSE
) {
  mlflow_restore_or_warning(restore)

  model_path <- resolve_model_path(model_path, run_id)

  if (!xor(is.null(input_path), is.null(data)))
    stop("One and only one of `input_path` or `data` must be specified.")

  data <- if (!is.null(input_path)) {
    switch(
      fs::path_ext(input_path),
      json = jsonlite::read_json(input_path),
      csv = read.csv(input_path)
    )
  } else {
    data
  }

  model <- mlflow_load_model(model_path)

  prediction <- mlflow_rfunc_predict_impl(model, data)

  if (is.null(output_path)) {
    if (!interactive()) message(prediction)

    prediction
  }
  else {
    switch(
      fs::path_ext(output_path),
      json = jsonlite::write_json(prediction, output_path),
      csv = write.csv(prediction, output_path, row.names = FALSE),
      stop("Unsupported output file format.")
    )
  }
}

resolve_model_path <- function(model_path, run_id) {
  if (!is.null(run_id)) {
    mlflow_get_or_create_active_connection()
    result <- withr::with_envvar(
      list(MLFLOW_TRACKING_URI = mlflow_tracking_uri()),
      mlflow_cli("artifacts", "download", "--run-id", run_id, "-a", model_path, echo = FALSE)
    )
      gsub("\n", "", result$stdout)
  } else {
    model_path
  }
}
