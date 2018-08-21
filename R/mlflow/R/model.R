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

  context_names <- names(formals(f))[2:length(formals(f))]

  context <- lapply(
    context_names,
    function(n) get0(n) %||% dynGet(n)
  )

  names(context) <- context_names

  model_raw <- serialize(
    list(
      r_function = f,
      context = context
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

mlflow_load_model <- function(model_dir) {
  spec <- yaml::read_yaml(fs::path(model_dir, "MLmodel"))

  if (!"r_function" %in% names(spec$flavors))
    stop("Model must define r_function to be used from R.")

  unserialize(readRDS(fs::path(model_dir, spec$flavors$r_function$model)))
}

mlflow_predict_model <- function(model, df) {
  do.call(model$r_function, args = model$context)
}

#' Predict using MLflow Model
#'
#' Predict using a MLflow Model from a 'JSON' file.
#'
#' @param model_dir The path to the MLflow model, as a string.
#' @param data Data frame, 'JSON' or 'CSV' file to be used for prediction.
#' @param output_file 'JSON' or 'CSV' file where the prediction will be written to.
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
  data,
  output_file = NULL,
  restore = FALSE
) {
  if (restore) mlflow_restore()

  if (is.character(data))
  {
    data <- switch(
      fs::path_ext(data),
      json = jsonlite::read_json(data),
      csv = read.csv(data)
    )
  }

  if (!is.data.frame(data))
    stop("Could not load data as a data frame.")

  model <- mlflow_load_model(model_dir)
  prediction <- mlflow_predict_model(model, data)

  if (is.null(output_file)) {
    if (!interactive()) message(prediction)

    prediction
  }
  else {
    switch(
      fs::path_ext(output_file),
      json = jsonlite::write_json(prediction, output_file),
      csv = write.csv(prediction, data_file, row.names = FALSE),
      stop("Unsupported output file format.")
    )
  }
}
