#' Save Model for MLflow
#'
#' Saves model in MLflow format that can later be used
#' for prediction and serving.
#'
#' @param x The serving function or model that will perform a prediction.
#' @param path Destination path where this MLflow compatible model
#'   will be saved.
#' @param r_dependencies Optional vector of paths to dependency files
#'   to include in the model, as in \code{r-dependencies.txt}
#'   or \code{conda.yaml}.
#' @param conda_env Path to Conda dependencies file.
#'
#' @importFrom yaml write_yaml
#' @export
mlflow_save_model <- function(x, path = "model", r_dependencies=NULL, conda_env=NULL) {

  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  flavor_spec <- list (
    flavors = mlflow_save_flavor(x, path, r_dependencies, conda_env)
  )
  mlflow_write_model_spec(path, flavor_spec)
}

#' Log Model
#'
#' Logs a model for this run. Similar to `mlflow_save_model()`
#' but stores model as an artifact within the active run.
#'
#' @param fn The serving function that will perform a prediction.
#' @param artifact_path Destination path where this MLflow compatible model
#'   will be saved.
#'
#' @export
mlflow_log_model <- function(fn, artifact_path) {
  temp_path <- fs::path_temp(artifact_path)
  mlflow_save_model(fn, path = temp_path)
  mlflow_log_artifact(path = temp_path, artifact_path = artifact_path)
}

mlflow_timestamp <- function() {
  withr::with_options(
    c(digits.secs = 2),
    format(
      as.POSIXlt(Sys.time(), tz = "GMT"),
      "%y-%m-%dT%H:%M:%S.%OS"
    )
  )
}

mlflow_write_model_spec <- function(path, content) {
  content$utc_time_created <- mlflow_timestamp()
  content$run_id <- mlflow_get_active_run_id()

  write_yaml(
    purrr::compact(content),
    file.path(path, "MLmodel")
  )
}

#' Generate Prediction with MLflow Model
#'
#' Generates a prediction with an MLflow model.
#'
#' @param model MLflow model.
#' @param data Dataframe to be scored.
#' @export
mlflow_predict_model <- function(model, data) {
   model %>% mlflow_predict_flavor(data)
}

#' Load MLflow Model
#'
#' Loads an MLflow model. MLflow models can have multiple model flavors. Not all flavors / models
#' can be loaded in R. This method by default searches for a flavor supported by R/MLflow.
#'
#' @param model_path Path to the MLflow model. The path is relative to the run with the given
#'        run-id or local filesystem path without run-id.
#' @param run_id Optional MLflow run-id. If supplied model will be fetched from MLflow tracking
#'        server.
#' @param flavor Optional flavor specification. Can be used to load a particular flavor in case
#'        there are multiple flavors available.
#' @export
mlflow_load_model <- function(model_path, flavor = NULL, run_id = NULL) {
  model_path <- resolve_model_path(model_path, run_id)
  supported_flavors <- supported_model_flavors()
  spec <- yaml::read_yaml(fs::path(model_path, "MLmodel"))
  available_flavors <- intersect(names(spec$flavors), supported_flavors)

  if (length(available_flavors) == 0) {
    stop(
      "Model does not contain any flavor supported by mlflow/R. ",
      "Model flavors: ",
      paste(names(spec$flavors), collapse = ", "),
      ". Supported flavors: ",
      paste(supported_flavors, collapse = ", "))
  }

  if (!is.null(flavor)) {
    if (!flavor %in% supported_flavors) {
      stop("Invalid flavor.", paste("Supported flavors:",
                              paste(supported_flavors, collapse = ", ")))
    }
    if (!flavor %in% available_flavors) {
      stop("Model does not contain requested flavor. ",
           paste("Available flavors:", paste(available_flavors, collapse = ", ")))
    }

    flavor <- flavor
  } else {
    if (length(available_flavors) > 1) {
      warning(paste("Multiple model flavors available (", paste(available_flavors, collapse = ", "),
                    " ).  loading flavor '", available_flavors[[1]], "'", ""))
    }

    flavor <- available_flavors[[1]]
  }

  flavor_path <- model_path
  class(flavor_path) <- c(flavor, class(flavor_path))
  mlflow_load_flavor(flavor_path)
}

#' Predict using RFunc MLflow Model
#'
#' Performs prediction using an RFunc MLflow model from a file or data frame.
#'
#' @param model_path The path to the MLflow model, as a string.
#' @param run_id Run ID of run to grab the model from.
#' @param input_path Path to 'JSON' or 'CSV' file to be used for prediction.
#' @param output_path 'JSON' or 'CSV' file where the prediction will be written to.
#' @param data Data frame to be scored. This can be used for testing purposes and can only
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

  prediction <- mlflow_predict_flavor(model, data)

  if (is.null(output_path)) {
    if (!interactive()) message(prediction)

    prediction
  } else {
    switch(
      fs::path_ext(output_path),
      json = jsonlite::write_json(prediction, output_path),
      csv = write.csv(prediction, output_path, row.names = FALSE),
      stop("Unsupported output file format.")
    )
  }
}

resolve_model_path <- function(model_path, run_id, client = mlflow_client()) {
  if (!is.null(run_id)) {
    result <- mlflow_cli("artifacts", "download", "--run-id", run_id, "-a", model_path,
                         echo = FALSE, client = client)
    gsub("\n", "", result$stdout)
  } else {
    model_path
  }
}

supported_model_flavors <- function() {
  purrr::map(utils::methods(generic.function = mlflow_load_flavor), ~ substring(.x, 20))
}
