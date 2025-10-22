#' @include model-utils.R
NULL

#' @rdname mlflow_save_model
#' @export
mlflow_save_model.xgb.Booster <- function(model,
                                          path,
                                          model_spec = list(),
                                          conda_env = NULL,
                                          ...) {
  assert_pkg_installed("xgboost")
  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  model_data_subpath <- "model.ubj"
  xgboost::xgb.save(model, fname = file.path(path, model_data_subpath))
  version <- remove_patch_version(
    as.character(utils::packageVersion("xgboost"))
  )

  pip_deps <- list("mlflow", paste("xgboost>=", version, sep = ""))
  conda_env <- create_default_conda_env_if_absent(path, conda_env, default_pip_deps = pip_deps)
  python_env <- create_python_env(path, dependencies = pip_deps)
  xgboost_conf <- list(
    xgboost = list(xgb_version = version, data = model_data_subpath)
  )
  pyfunc_conf <- create_pyfunc_conf(
    loader_module = "mlflow.xgboost",
    data = model_data_subpath,
    env = list(conda = conda_env, virtualenv = python_env)
  )
  model_spec$flavors <- append(append(model_spec$flavors, xgboost_conf), pyfunc_conf)

  mlflow_write_model_spec(path, model_spec)
}

#' @export
mlflow_load_flavor.mlflow_flavor_xgboost <- function(flavor, model_path) {
  assert_pkg_installed("xgboost")
  model_data_subpath <- "model.ubj"
  xgboost::xgb.load(file.path(model_path, model_data_subpath))
}

#' @export
mlflow_predict.xgb.Booster <- function(model, data, ...) {
  assert_pkg_installed("xgboost")
  stats::predict(model, xgboost::xgb.DMatrix(as.matrix(data)), ...)
}
