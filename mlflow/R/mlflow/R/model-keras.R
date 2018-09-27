#' Save MLflow Keras Model Flavor
#'
#' Saves model in MLflow's Keras flavor.
#'
#' @param x The serving function or model that will perform a prediction.
#' @param path Destination path where this MLflow compatible model
#'   will be saved.
#' @param r_dependencies Optional vector of paths to dependency files
#'   to include in the model, as in \code{r-dependencies.txt}
#'   or \code{conda.yaml}.
#' @param conda_env Path to Conda dependencies file.
#'
#' @return This funciton must return a list of flavors that conform to
#'   the MLmodel specification.
#'
#' @export
mlflow_save_flavor.keras.engine.training.Model <- function(x,
                                                           path = "model",
                                                           r_dependencies=NULL,
                                                           conda_env=NULL) {
  if (!"package:keras" %in% search()) {
    stop("The 'keras' package is not available, use 'library(keras)' before exporting a model.")
  }
  save_model_hdf5 <- get("save_model_hdf5", envir = as.environment("package:keras"))

  save_model_hdf5(x, filepath = file.path(path, "model.h5"), include_optimizer = TRUE)
  version <- as.character(utils::packageVersion("keras"))
  conda_env <- if (!is.null(conda_env)) {
    dst <- file.path(path, basename(conda_env))
    if (conda_env != dst) {
      file.copy(from = conda_env, to = dst)
    }
    basename(conda_env)
  } else { # create default conda environment
    conda_deps <- list()
    pip_deps <- list("mlflow", paste("keras>=", version, sep = ""))
    create_conda_env(name = "conda_env",
                     path = file.path(path, "conda_env.yaml"),
                     conda_deps = conda_deps,
                     pip_deps = pip_deps)
    "conda_env.yaml"
  }

  keras_conf <- list(
    keras = list(
      version = "2.2.2",
      data = "model.h5"))

  pyfunc_conf <- create_pyfunc_conf(
    loader_module = "mlflow.keras",
    data = "model.h5",
    env = conda_env)

  append(keras_conf, pyfunc_conf)
}

#' @export
mlflow_load_flavor.keras <- function(model_path) {
  if (!"package:keras" %in% search()) {
    stop("The 'keras' package is not available, use 'library(keras)' before loading a model.")
  }

  load_model_hdf5 <- get("load_model_hdf5", envir = as.environment("package:keras"))
  load_model_hdf5(file.path(model_path, "model.h5"))
}

#' @export
mlflow_predict_flavor.keras.engine.training.Model <- function(model, data) {
  if (!"package:keras" %in% search()) {
    stop("The 'keras' package is not available, use 'library(keras)' before calling predict.")
  }

  stats::predict(model, as.matrix(data))
}
