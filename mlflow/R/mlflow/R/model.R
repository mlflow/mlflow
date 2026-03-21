#' Save Model for MLflow
#'
#' Saves model in MLflow format that can later be used for prediction and serving. This method is
#' generic to allow package authors to save custom model types.
#'
#' @param model The model that will perform a prediction.
#' @param path Destination path where this MLflow compatible model
#'   will be saved.
#' @param model_spec MLflow model config this model flavor is being added to.
#' @param ... Optional additional arguments.
#' @importFrom yaml write_yaml
#' @export
mlflow_save_model <- function(model, path, model_spec = list(), ...) {
  UseMethod("mlflow_save_model")
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
  model_spec <- mlflow_save_model(model, path = temp_path, model_spec = list(
    utc_time_created = mlflow_timestamp(),
    run_id = mlflow_get_active_run_id_or_start_run(),
    artifact_path = artifact_path,
    flavors = list()
  ), ...)
  res <- mlflow_log_artifact(path = temp_path, artifact_path = artifact_path)
  tryCatch({ mlflow_record_logged_model(model_spec) }, error = function(e) {
    warning(paste("Logging model metadata to the tracking server has failed, possibly due to older",
                  "server version. The model artifacts have been logged successfully.",
                  "In addition to exporting model artifacts, MLflow clients 1.7.0 and above",
                  "attempt to record model metadata to the  tracking store. If logging to a",
                  "mlflow server via REST, consider  upgrading the server version to MLflow",
                  "1.7.0 or above.", sep=" "))
  })
  res
}

mlflow_write_model_spec <- function(path, content) {
  write_yaml(
    purrr::compact(content),
    file.path(path, "MLmodel")
  )
}

mlflow_timestamp <- function() {
  withr::with_options(
    c(digits.secs = 2),
    format(
      as.POSIXlt(Sys.time(), tz = "GMT"),
      "%Y-%m-%d %H:%M:%OS6"
    )
  )
}

#' Load MLflow Model
#'
#' Loads an MLflow model. MLflow models can have multiple model flavors. Not all flavors / models
#' can be loaded in R. This method by default searches for a flavor supported by R/MLflow.
#'
#' @template roxlate-model-uri
#' @template roxlate-client
#' @param flavor Optional flavor specification (string). Can be used to load a particular flavor in
#' case there are multiple flavors available.
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
  } else {
    if (length(available_flavors) > 1) {
      warning(paste("Multiple model flavors available (", paste(available_flavors, collapse = ", "),
                    " ).  loading flavor '", available_flavors[[1]], "'", ""))
    }
    flavor <- available_flavors[[1]]
  }

  flavor <- mlflow_flavor(flavor, spec$flavors[[flavor]])
  mlflow_load_flavor(flavor, model_path)
}

new_mlflow_flavor <- function(class = character(0), spec = NULL) {
  flavor <- structure(character(0), class = c(class, "mlflow_flavor"))
  attributes(flavor)$spec <- spec

  flavor
}

# Create an MLflow Flavor Object
#
# This function creates an `mlflow_flavor` object that can be used to dispatch
#   the `mlflow_load_flavor()` method.
#
# @param flavor The name of the flavor.
# @keywords internal
mlflow_flavor <- function(flavor, spec) {
  new_mlflow_flavor(class = paste0("mlflow_flavor_", flavor), spec = spec)
}

#' Load MLflow Model Flavor
#'
#' Loads an MLflow model using a specific flavor. This method is called internally by
#' \link[mlflow]{mlflow_load_model}, but is exposed for package authors to extend the supported
#' MLflow models. See https://mlflow.org/docs/latest/models.html#storage-format for more
#' info on MLflow model flavors.
#'
#' @param flavor An MLflow flavor object loaded by \link[mlflow]{mlflow_load_model}, with class
#' loaded from the flavor field in an MLmodel file.
#' @param model_path The path to the MLflow model wrapped in the correct
#'   class.
#'
#' @export
mlflow_load_flavor <- function(flavor, model_path) {
  UseMethod("mlflow_load_flavor")
}

#' Generate Prediction with MLflow Model
#'
#' Performs prediction over a model loaded using
#' \code{mlflow_load_model()}, to be used by package authors
#' to extend the supported MLflow models.
#'
#' @param model The loaded MLflow model flavor.
#' @param data A data frame to perform scoring.
#' @param ... Optional additional arguments passed to underlying predict
#'   methods.
#'
#' @export
mlflow_predict <- function(model, data, ...) {
  UseMethod("mlflow_predict")
}


# Generate predictions using a saved R MLflow model.
# Input and output are read from and written to a specified input / output file or stdin / stdout.
#
# @param input_path Path to 'JSON' or 'CSV' file to be used for prediction. If not specified data is
#                   read from the stdin.
# @param output_path 'JSON' file where the prediction will be written to. If not specified,
#                     data is written out to stdout.

mlflow_rfunc_predict <- function(model_path, input_path = NULL, output_path = NULL,
                                 content_type = NULL) {
  model <- mlflow_load_model(model_path)
  input_path <- input_path %||% "stdin"
  output_path <- output_path %||% stdout()

  data <- switch(
    content_type %||% "json",
    json = parse_json(input_path),
    csv = utils::read.csv(input_path),
    stop("Unsupported input file format.")
  )
  model <- mlflow_load_model(model_path)
  prediction <- mlflow_predict(model, data)
  jsonlite::write_json(prediction, output_path, digits = NA)
  invisible(NULL)
}

supported_model_flavors <- function() {
  purrr::map(utils::methods(generic.function = mlflow_load_flavor),
             ~ gsub("mlflow_load_flavor\\.mlflow_flavor_", "", .x))
}

# Helper function to parse data frame from json.
parse_json <- function(input_path) {
  json <- jsonlite::fromJSON(input_path, simplifyVector = TRUE)
  data_fields <- intersect(names(json), c("dataframe_split", "dataframe_records"))
  if (length(data_fields) != 1) {
    stop(paste(
      "Invalid input. The input must contain 'dataframe_split' or 'dataframe_records' but not both.",
      "Got input fields", names(json))
    )
  }
  switch(data_fields[[1]],
    dataframe_split = {
      elms <- names(json$dataframe_split)
      if (length(setdiff(elms, c("columns", "index", "data"))) != 0
      || length(setdiff(c("data"), elms) != 0)) {
        stop(paste("Invalid input. Make sure the input json data is in 'split' format.", elms))
      }
      data <- if (any(class(json$dataframe_split$data) == "list")) {
        max_len <- max(sapply(json$dataframe_split$data, length))
        fill_nas <- function(row) {
          append(row, rep(NA, max_len - length(row)))
        }
        rows <- lapply(json$dataframe_split$data, fill_nas)
        Reduce(rbind, rows)
      } else {
        json$dataframe_split$data
      }

      df <- data.frame(data, row.names=json$dataframe_split$index)
      names(df) <- json$dataframe_split$columns
      df
    },
    dataframe_records = json$dataframe_records,
    stop(paste("Unsupported JSON format"))
  )
}
