#' Save Model for MLflow
#'
#' Saves model in MLflow format that can later be used
#' for prediction and serving.
#'
#' @param model The model that will perform a prediction.
#' @param path Destination path where this MLflow compatible model
#'   will be saved.
#' @param ... Optional additional arguments passed to `mlflow_save_flavor()` - for example,
#'   `conda_env = /path/to/conda.yaml` may be passed to specify a conda dependencies file
#'   for flavors (e.g. keras) that support conda environments.
#' @importFrom yaml write_yaml
#' @export
mlflow_save_model <- function(model, path = "model", ...) {
  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  flavor_spec <- list (
    flavors = mlflow_save_flavor(model, path, ...)
  )
  mlflow_write_model_spec(path, flavor_spec)
}

#' Log Model
#'
#' Logs a model for this run. Similar to `mlflow_save_model()`
#' but stores model as an artifact within the active run.
#'
#' @param model The model that will perform a prediction.
#' @param artifact_path Destination path where this MLflow compatible model
#'   will be saved.
#' @param ... Optional additional arguments passed to `mlflow_save_model()` when persisting the
#'   model. For example, `conda_env = /path/to/conda.yaml` may be passed to specify a conda
#'   dependencies file for flavors (e.g. keras) that support conda environments.
#'
#' @export
mlflow_log_model <- function(model, artifact_path, ...) {
  temp_path <- fs::path_temp(artifact_path)
  mlflow_save_model(model, path = temp_path, ...)
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
#' @template roxlate-model-uri
#' @template roxlate-client
#' @param flavor Optional flavor specification. Can be used to load a particular flavor in case
#'        there are multiple flavors available.
#' @export
mlflow_load_model <- function(model_uri, flavor = NULL, client = mlflow_client()) {
  model_path <- mlflow_download_artifacts_from_uri(model_uri, client = client)
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


# Generate predictions using a saved R MLflow model.
# Input and output are read from and written to a specified input / output file or stdin / stdout.
#
# @param input_path Path to 'JSON' or 'CSV' file to be used for prediction. If not specified data is
#                   read from the stdin.
# @param output_path 'JSON' file where the prediction will be written to. If not specified,
#                     data is written out to stdout.

mlflow_rfunc_predict <- function(model_path, input_path = NULL, output_path = NULL,
                                 content_type = NULL, json_format = NULL) {
  model <- mlflow_load_model(model_path)
  input_path <- input_path %||% "stdin"
  output_path <- output_path %||% stdout()

  data <- switch(
    content_type %||% "json",
    json = parse_json(input_path, json_format %||% "split"),
    csv = read.csv(input_path),
    stop("Unsupported input file format.")
  )
  model <- mlflow_load_model(model_path)
  prediction <- mlflow_predict_flavor(model, data)
  jsonlite::write_json(prediction, output_path, digits = NA)
  invisible(NULL)
}

supported_model_flavors <- function() {
  purrr::map(utils::methods(generic.function = mlflow_load_flavor), ~ substring(.x, 20))
}

# Helper function to parse data frame from json based on given the json_fomat.
# The default behavior is to parse the data in the Pandas "split" orient.
parse_json <- function(input_path, json_format="split") {
  switch(json_format,
    split = {
      json <- jsonlite::read_json(input_path, simplifyVector = TRUE)
      elms <- names(json)
      if (length(setdiff(elms, c("columns", "index", "data"))) != 0
      || length(setdiff(c("columns", "data"), elms) != 0)) {
        stop(paste("Invalid input. Make sure the input json data is in 'split' format.", elms))
      }
      df <- data.frame(json$data, row.names = json$index)
      names(df) <- json$columns
      df
    },
    records = jsonlite::read_json(input_path, simplifyVector = TRUE),
    stop(paste("Unsupported JSON format", json_format,
               ". Supported formats are 'split' or 'records'"))
  )
}
