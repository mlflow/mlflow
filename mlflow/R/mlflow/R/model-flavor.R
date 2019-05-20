#' Load MLflow Model Flavor
#'
#' Loads an MLflow model flavor, to be used by package authors
#' to extend the supported MLflow models.
#'
#' @param model_path The path to the MLflow model wrapped in the correct
#'   class.
#'
#' @export
mlflow_load_flavor <- function(model_path) {
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
