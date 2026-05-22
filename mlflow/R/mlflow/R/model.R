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

mlflow_signature_supported_types <- c(
  "boolean", "integer", "long", "float", "double", "string", "binary", "datetime",
  "array", "object", "map", "any", "sparkml_vector"
)

mlflow_signature_scalar_string <- function(value) {
  is.character(value) && length(value) == 1 && !is.na(value) && nzchar(value)
}

mlflow_normalize_signature_node <- function(node, path, default_required = FALSE) {
  if (mlflow_signature_scalar_string(node)) {
    node <- list(type = as.character(node))
  }

  if (is.data.frame(node) || !is.list(node)) {
    stop(
      sprintf("Signature field `%s` must be a scalar type string or a list with `type`.", path),
      call. = FALSE
    )
  }

  if (!mlflow_signature_scalar_string(node$type)) {
    stop(sprintf("Signature field `%s` must include a scalar `type`.", path), call. = FALSE)
  }

  node$type <- as.character(node$type)
  if (!node$type %in% mlflow_signature_supported_types) {
    stop(sprintf("Unsupported signature type `%s` in `%s`.", node$type, path), call. = FALSE)
  }

  if (!is.null(node$required)) {
    valid_required <- is.logical(node$required) &&
      length(node$required) == 1 &&
      !is.na(node$required)
    if (!valid_required) {
      stop(sprintf("Signature field `%s` has invalid `required` value.", path), call. = FALSE)
    }
  } else if (default_required) {
    node$required <- TRUE
  }

  if (identical(node$type, "array")) {
    if (is.null(node$items)) {
      stop(sprintf("Signature array `%s` must include `items`.", path), call. = FALSE)
    }
    node$items <- mlflow_normalize_signature_node(node$items, paste0(path, ".items"))
  } else if (identical(node$type, "object")) {
    node$properties <- mlflow_normalize_signature_properties(node$properties, path)
  } else if (identical(node$type, "map")) {
    if (is.null(node$values)) {
      stop(sprintf("Signature map `%s` must include `values`.", path), call. = FALSE)
    }
    node$values <- mlflow_normalize_signature_node(node$values, paste0(path, ".values"))
  }

  node
}

mlflow_normalize_signature_properties <- function(properties, path) {
  if (
    is.null(properties) || is.data.frame(properties) || !is.list(properties) ||
      length(properties) == 0 || is.null(names(properties)) || any(!nzchar(names(properties)))
  ) {
    stop(sprintf("Signature object `%s` must include named `properties`.", path), call. = FALSE)
  }

  purrr::imap(properties, function(property, name) {
    mlflow_normalize_signature_node(
      property,
      paste0(path, ".properties.", name),
      default_required = TRUE
    )
  })
}

mlflow_normalize_signature_field <- function(field, name, field_set) {
  field <- mlflow_normalize_signature_node(
    field,
    paste0(field_set, ".", name),
    default_required = TRUE
  )
  c(list(name = as.character(name)), field[setdiff(names(field), "name")])
}

mlflow_normalize_signature_fields <- function(fields, field_set) {
  if (is.null(fields)) return(NULL)
  if (is.data.frame(fields)) {
    stop(
      sprintf("Signature `%s` must be a named list, not a data.frame.", field_set),
      call. = FALSE
    )
  }
  if (
    !is.list(fields) || length(fields) == 0 ||
      is.null(names(fields)) || any(!nzchar(names(fields)))
  ) {
    stop(
      sprintf("Signature `%s` must be a named list, e.g. list(feature = \"double\").", field_set),
      call. = FALSE
    )
  }
  unname(purrr::imap(fields, ~ mlflow_normalize_signature_field(.x, .y, field_set)))
}

mlflow_signature_schema_json <- function(fields) {
  if (is.null(fields)) return(NULL)
  as.character(jsonlite::toJSON(fields, auto_unbox = TRUE))
}

mlflow_normalize_signature <- function(signature) {
  if (is.null(signature)) return(NULL)
  if (is.data.frame(signature)) {
    stop("`signature` must be a named list, not a data.frame.", call. = FALSE)
  }
  if (!is.list(signature)) {
    stop("`signature` must be a list with `inputs` and `outputs`.", call. = FALSE)
  }
  normalized <- list(
    inputs = mlflow_signature_schema_json(
      mlflow_normalize_signature_fields(signature$inputs, "inputs")
    ),
    outputs = mlflow_signature_schema_json(
      mlflow_normalize_signature_fields(signature$outputs, "outputs")
    )
  )
  if (is.null(normalized$inputs) && is.null(normalized$outputs)) {
    stop("`signature` must include `inputs` or `outputs`.", call. = FALSE)
  }
  normalized
}

#' Log Model
#'
#' Logs a model for this run. Similar to `mlflow_save_model()`
#' but stores model as an artifact within the active run.
#'
#' @param model The model that will perform a prediction.
#' @param artifact_path Destination path where this MLflow compatible model
#'   will be saved.
#' @param signature Optional model signature with named-list `inputs` and `outputs`,
#'   e.g. `list(inputs = list(feature = "double"), outputs = list(prediction = "double"))`.
#' @param ... Optional additional arguments passed to `mlflow_save_model()` when persisting the
#'   model. For example, `conda_env = /path/to/conda.yaml` may be passed to specify a conda
#'   dependencies file for flavors (e.g. keras) that support conda environments.
#'
#' @export
mlflow_log_model <- function(model, artifact_path, signature = NULL, ...) {
  temp_path <- fs::path_temp(artifact_path)
  signature <- mlflow_normalize_signature(signature)
  model_spec <- list(
    utc_time_created = mlflow_timestamp(),
    run_id = mlflow_get_active_run_id_or_start_run(),
    artifact_path = artifact_path,
    flavors = list()
  )
  if (!is.null(signature)) {
    model_spec$signature <- signature
  }
  model_spec <- mlflow_save_model(model, path = temp_path, model_spec = model_spec, ...)
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

mlflow_is_plain_models_uri <- function(model_uri) {
  grepl("^models:/[^/]", model_uri)
}

mlflow_parse_models_uri <- function(model_uri) {
  rest <- sub("^models:/", "", model_uri)
  parts <- strsplit(rest, "/", fixed = TRUE)[[1]]
  if (!nzchar(rest) || length(parts) > 2 || !nzchar(parts[[1]])) {
    stop(
      "Model URIs must be of the form `models:/name/version` or `models:/name@alias`.",
      call. = FALSE
    )
  }

  if (length(parts) == 2) {
    suffix <- parts[[2]]
    if (!nzchar(suffix)) {
      stop(
        "Model URIs must include a non-empty version, stage, or alias.",
        call. = FALSE
      )
    }
    return(list(
      name = parts[[1]],
      version = if (grepl("^[0-9]+$", suffix)) suffix else NULL,
      stage = if (grepl("^[0-9]+$", suffix)) NULL else suffix,
      alias = NULL
    ))
  }

  alias_pos <- gregexpr("@", rest, fixed = TRUE)[[1]]
  if (alias_pos[[1]] != -1L) {
    at <- alias_pos[[length(alias_pos)]]
    name <- substr(rest, 1, at - 1)
    alias <- substr(rest, at + 1, nchar(rest))
    if (!nzchar(name) || !nzchar(alias)) {
      stop(
        "Model alias URIs must be of the form `models:/name@alias`.",
        call. = FALSE
      )
    }
    return(list(name = name, version = NULL, stage = NULL, alias = alias))
  }

  list(name = rest, version = NULL, stage = NULL, alias = NULL)
}

mlflow_resolve_model_uri_version <- function(parsed, client) {
  if (is.null(parsed$alias)) {
    return(parsed$version)
  }

  alias_resp <- mlflow_get_model_version_by_alias(parsed$name, parsed$alias, client = client)
  mv <- alias_resp$model_version %||% alias_resp
  if (is.null(mv$version) || !nchar(mv$version)) {
    stop("Unable to resolve model alias to a concrete model version.", call. = FALSE)
  }
  mv$version
}

mlflow_download_model_uri <- function(model_uri, client) {
  parsed <- mlflow_parse_models_uri(model_uri)
  version <- mlflow_resolve_model_uri_version(parsed, client)

  if (is_uc_registry_uri(client)) {
    if (!is.null(parsed$stage)) {
      mlflow_uc_stage_error("mlflow_load_model")
    }
    if (is.null(version) || !nchar(version)) {
      stop("Unity Catalog model URIs must include a concrete version or alias.", call. = FALSE)
    }
    return(mlflow_download_uc_model_version(parsed$name, version, client = client))
  }

  if (is.null(parsed$alias)) {
    return(NULL)
  }

  resolved_uri <- mlflow_get_model_version_download_uri(parsed$name, version, client = client)
  mlflow_download_artifacts_from_uri(resolved_uri, client = client)
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
  model_path <- NULL
  if (mlflow_is_plain_models_uri(model_uri)) {
    model_path <- mlflow_download_model_uri(model_uri, client = client)
  }

  model_path <- model_path %||% mlflow_download_artifacts_from_uri(model_uri, client = client)
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
