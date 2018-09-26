#' importFrom mlflow create_conda_env, create_pyfunc_conf
#' @export
mlflow_save_flavor.keras.engine.training.Model <- function(x,
                                                           path = "model",
                                                           r_dependencies=NULL,
                                                           conda_env=NULL) {
  save_model_hdf5(x, filepath = file.path(path, "model.h5"), include_optimizer = TRUE)
  version <- as.character(packageVersion("keras"))
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
  # verify that Keras is installed
  result <- tryCatch({
    packageVersion("keras")
  }, error = function(e) {
    if (e$message == "package ‘keras’ not found"){
      stop("Keras package is needed to load this model.")
    }
    stop(e)
  })
  load_model_hdf5(file.path(model_path, "model.h5"))
}

#' @export
mlflow_predict_flavor.keras.engine.training.Model <- function(model, data) {
  predict(model, as.matrix(data))
}
