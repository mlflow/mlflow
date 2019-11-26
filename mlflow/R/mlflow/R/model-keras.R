#' @rdname mlflow_save_model
#' @param conda_env Path to Conda dependencies file.
#' @export
mlflow_save_model.keras.engine.training.Model <- function(model,
                                                          path,
                                                          conda_env = NULL,
                                                          ...) {
  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  if (!requireNamespace("keras", quietly = TRUE)) {
    stop("The 'keras' package must be installed.")
  }

  keras::save_model_hdf5(model, filepath = file.path(path, "model.h5"), include_optimizer = TRUE)
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

  mlflow_write_model_spec(path, list(
    flavors = append(keras_conf, pyfunc_conf)
  ))
}

#' @export
mlflow_load_flavor.mlflow_flavor_keras <- function(flavor, model_path) {
  if (!requireNamespace("keras", quietly = TRUE)) {
    stop("The 'keras' package must be installed.")
  }

  keras::load_model_hdf5(file.path(model_path, "model.h5"))
}

#' @export
mlflow_predict.keras.engine.training.Model <- function(model, data, ...) {
  if (!requireNamespace("keras", quietly = TRUE)) {
    stop("The 'keras' package must be installed.")
  }

  stats::predict(model, as.matrix(data), ...)
}
