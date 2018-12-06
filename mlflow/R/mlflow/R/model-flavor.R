#' Save MLflow Model Flavor
#'
#' Saves model in MLflow flavor, to be used by package authors
#' to extend the supported MLflow models.
#'
#' @param x The serving function or model that will perform a prediction.
#' @param path Destination path where this MLflow compatible model
#'   will be saved.
#' @param r_dependencies Optional vector of paths to dependency files
#'   to include in the model, as in \code{r-dependencies.txt}
#'   or \code{conda.yaml}.
#' @param conda_env Path to Conda dependencies file.
#'
#' @return This function must return a list of flavors that conform to
#'   the MLmodel specification.
#'
#' @export
mlflow_save_flavor <- function(x, path = "model", r_dependencies=NULL, conda_env=NULL) {
  UseMethod("mlflow_save_flavor")
}

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

#' Predict over MLflow Model Flavor
#'
#' Performs prediction over a model loaded using
#' \code{mlflow_load_model()}, to be used by package authors
#' to extend the supported MLflow models.
#'
#' @param model The loaded MLflow model flavor.
#' @param data A data frame to perform scoring.
#'
#' @export
mlflow_predict_flavor <- function(model, data) {
  UseMethod("mlflow_predict_flavor")
}
