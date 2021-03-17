#' @include model-utils.R
NULL

# TODO: currently the equivalent of mlflow_save_model for the workflow flavor is
# implemented in `containerize`
#
# mlflow_save_flavor.workflow <- function(...

#' @export
mlflow_load_flavor.mlflow_flavor_workflow <- function(flavor, model_path) {
  assert_pkg_installed("workflows")
  model_object_file <- attributes(flavor)$spec$model_object_file
  if (is.null(model_object_file)) {
    stop("'model_object_file' attribute is missing")
  }

  readRDS(file.path(model_path, model_object_file))
}

#' @export
mlflow_predict.workflow <- function(model, data, ...) {
  predict(model, data)$.pred
}
