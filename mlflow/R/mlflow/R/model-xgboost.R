#' @rdname mlflow_save_model
#' @export
mlflow_save_model.xgb.Booster <- function(model,
                                          path,
                                          model_spec = list(),
                                          conda_env = NULL,
                                          ...) {
  xgboost_assert_installed()
  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  model_data_subpath <- "model.xgb"
  xgboost::xgb.save(model, fname = file.path(path, model_data_subpath))
  version <- as.character(utils::packageVersion("xgboost"))
  # needed because R and python packages don't have same patch number
  version <- gsub("([^.]*\\.[^.]*)(\\..*)", "\\1", version)

  conda_env <- if (!is.null(conda_env)) {
    dst <- file.path(path, basename(conda_env))
    if (conda_env != dst) {
      file.copy(from = conda_env, to = dst)
    }
    basename(conda_env)
  } else { # create default conda environment
    conda_deps <- list()
    pip_deps <- list("mlflow", paste("xgboost>=", version, sep = ""))
    create_conda_env(
      name = "conda_env",
      path = file.path(path, "conda_env.yaml"),
      conda_deps = conda_deps,
      pip_deps = pip_deps
    )
    "conda_env.yaml"
  }

  xgboost_conf <- list(
    xgboost = list(xgb_version = version, data = model_data_subpath)
  )

  pyfunc_conf <- create_pyfunc_conf(
    loader_module = "mlflow.xgboost",
    data = model_data_subpath,
    env = conda_env
  )
  model_spec$flavors <- append(append(model_spec$flavors, xgboost_conf), pyfunc_conf)
  mlflow_write_model_spec(path, model_spec)
}

#' @export
mlflow_load_flavor.mlflow_flavor_xgboost <- function(flavor, model_path) {
  xgboost_assert_installed()
  model_data_subpath <- "model.xgb"
  xgboost::xgb.load(file.path(model_path, model_data_subpath))
}

#' @export
mlflow_predict.xgb.Booster <- function(model, data, ...) {
  xgboost_assert_installed()
  stats::predict(model, xgboost::xgb.DMatrix(as.matrix(data)), ...)
}

xgboost_assert_installed <- function() {
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("The 'xgboost' package must be installed.")
  }
}
