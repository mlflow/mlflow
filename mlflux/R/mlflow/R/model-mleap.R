#' @include model-utils.R
NULL

#' @rdname mlflow_save_model
#' @param sample_input Sample Spark DataFrame input that the model can evaluate. This is required by MLeap for data schema inference.
#'
#' @export
mlflow_save_model.ml_pipeline_model <- function(model,
                                                path,
                                                model_spec = list(),
                                                conda_env = NULL,
                                                sample_input = NULL,
                                                ...) {
  if (is.null(sample_input)) {
    stop("`sample_input` is required by MLeap for data schema inference.")
  }

  assert_pkg_installed("mleap")

  model_filename <- "model.zip"

  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)
  model_path <- file.path(path, model_filename)
  mleap::ml_write_bundle(model, sample_input = sample_input, path = model_path)
  version <- mleap::mleap_installed_versions()$mleap %>%
    purrr::map(~ numeric_version(.x)) %>%
    purrr::reduce(~ (if (.x > .y) .x else .y)) %>%
    as.character()

  conda_env <- create_default_conda_env_if_absent(
    path, conda_env, default_pip_deps = list("mlflow", paste("mleap>=", version, sep = ""))
  )
  mleap_conf <- list(
    mleap = list(mleap_version = version, model_data = model_filename)
  )
  model_spec$flavors <- append(model_spec$flavors, mleap_conf)


  mlflow_write_model_spec(path, model_spec)
}

#' @export
mlflow_load_flavor.mlflow_flavor_mleap <- function(flavor, model_path) {
  assert_pkg_installed("mleap")
  model_data <- attributes(flavor)$spec$model_data
  if (is.null(model_data)) {
    stop("'model_data' attribute is missing")
  }
  mleap::mleap_load_bundle(file.path(model_path, model_data))
}

#' @export
mlflow_predict.mleap_transformer <- function(model, data, ...) {
  assert_pkg_installed("mleap")
  mleap::mleap_transform(model, data, ...)
}
