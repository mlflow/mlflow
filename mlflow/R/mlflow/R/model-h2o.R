#' @include model-utils.R
NULL

#' @rdname mlflow_save_model
#' @export
mlflow_save_model.H2OModel <- function(model,
                                       path,
                                       model_spec = list(),
                                       conda_env = NULL,
                                       ...) {
  assert_pkg_installed("h2o")

  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path, recursive = TRUE)
  path <- normalizePath(path)

  model_data_subpath <- "model.h2o"
  model_data_path <- file.path(path, model_data_subpath)

  dir.create(model_data_path)

  h2o_save_location <- h2o::h2o.saveModel(
    object = model, path = model_data_path, force = TRUE
  )
  model_file <- basename(h2o_save_location)

  settings <- list(
    full_file = h2o_save_location,
    model_file = model_file,
    model_dir = model_data_path
  )
  yaml::write_yaml(settings, file.path(model_data_path, "h2o.yaml"))

  pip_deps <- list("mlflow", paste0("h2o==", as.character(utils::packageVersion("h2o"))))
  conda_env <- create_default_conda_env_if_absent(path, conda_env, default_pip_deps = pip_deps)
  python_env <- create_python_env(path, dependencies = pip_deps)

  h2o_conf <- list(
    h2o = list(h2o_version = version, data = model_data_subpath)
  )
  pyfunc_conf <- create_pyfunc_conf(
    loader_module = "mlflow.h2o",
    data = model_data_subpath,
    env = list(conda = conda_env, virtualenv = python_env)
  )
  model_spec$flavors <- c(model_spec$flavors, h2o_conf, pyfunc_conf)
  mlflow_write_model_spec(path, model_spec)
}

#' @importFrom rlang %||%
#' @export
mlflow_load_flavor.mlflow_flavor_h2o <- function(flavor, model_path) {
  assert_pkg_installed("h2o")

  model_path <- normalizePath(model_path)
  # Flavor configurations for models saved in MLflow version <= 0.8.0 may not contain a
  # `data` key; in this case, we assume the model artifact path to be `model.h2o
  model_data_subpath <- attributes(flavor)$spec$data %||% "model.h2o"

  h2o_model_file_path <- file.path(model_path, model_data_subpath)
  settings <- yaml::read_yaml(file.path(h2o_model_file_path, "h2o.yaml"))
  h2o::h2o.loadModel(file.path(h2o_model_file_path, settings$model_file))
}

#' @export
mlflow_predict.H2OModel <- function(model, data, ...) {
  assert_pkg_installed("h2o")
  as.data.frame(h2o::h2o.predict(model, h2o::as.h2o(data), ...))
}
