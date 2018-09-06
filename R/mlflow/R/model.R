#' Save Model for MLflow
#'
#' Saves model in MLflow's format that can later be used
#' for prediction and serving.
#'
#' @param x The serving function or model that will perform a prediction.
#' @param path Destination path where this MLflow compatible model
#'   will be saved.
#' @param dependencies Optional vector of paths to dependency files
#'   to include in the model, as in \code{r-dependencies.txt}
#'   or \code{conda.yaml}.
#'
#' @importFrom yaml write_yaml
#' @export
mlflow_save_model <- function(x, path = "model", dependencies = NULL) {
  supported_deps <- list(
    "r-dependencies.txt" = "r_dependencies",
    "conda.yaml" = "conda_env"
  )

  if (!is.null(dependencies)) {
    dep_file <- basename(dependencies)
    if (!all(dep_file %in% names(supported_deps))) {
      stop("Dependency ", dep_file, " is unsupported. Supported files: ", paste(names(supported_deps), collapse = ", "))
    }
  }

  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  flavor_spec <- list (
    flavors = mlflow_save_flavor(x, path)
  )

  for (dependency in dependencies) {
    flavor_spec[[supported_deps[[basename(dependency)]]]] <- dependency
  }

  mlflow_write_model_spec(path, flavor_spec)
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
  content$time_created <- mlflow_timestamp()
  content$run_id <- mlflow_active_run()$run_info$run_uuid

  write_yaml(
    purrr::compact(content),
    file.path(path, "MLmodel")
  )
}

mlflow_load_model <- function(model_path) {
  spec <- yaml::read_yaml(fs::path(model_path, "MLmodel"))

  model_flavors <- gsub("^r_", "", names(spec$flavors))

  supported <- model_flavors %>%
    Filter(function(e) paste("mlflow_load_flavor", e, sep = ".") %in% as.vector(methods(class = e)), .)

  if ("crate" %in% model_flavors) {
    supported <- c("crate", supported)
  }

  if (length(supported) == 0) {
    stop(
      "Model must define r_crate flavor to be used from R. ",
      "Model flavors: ",
      paste(model_flavors, collapse = ", "),
      ". Supported flavors: ",
      paste(
        purrr::map_chr(model_flavors, ~ as.vector(methods(class = .x))),
        collapse = ", "
      )
    )
  }

  supported_class <- supported[[1]]

  flavor_path <- fs::path(model_path, spec$flavors[[paste("r", supported_class, sep = "_")]]$model)
  class(flavor_path) <- c(supported_class, class(flavor_path))

  mlflow_load_flavor(flavor_path)
}

#' Predict using RFunc MLflow Model
#'
#' Predict using an RFunc MLflow Model from a file or data frame.
#'
#' @param model_path The path to the MLflow model, as a string.
#' @param run_uuid Run ID of run to grab the model from.
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
  run_uuid = NULL,
  input_path = NULL,
  output_path = NULL,
  data = NULL,
  restore = FALSE
) {
  mlflow_restore_or_warning(restore)

  model_path <- resolve_model_path(model_path, run_uuid)

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

resolve_model_path <- function(model_path, run_uuid) {
  if (!is.null(run_uuid)) {
    mlflow_get_or_create_active_connection()
    result <- mlflow_cli("artifacts", "download", "--run-id", run_uuid, "-a", model_path, echo = FALSE)

    gsub("\n", "", result$stdout)
  } else {
    model_path
  }
}
