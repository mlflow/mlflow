# `trivial` is a fictional MLflow flavor that exists only for unit test purposes

mlflow_save_model.trivial <- function(model, path, model_spec = list(), ...) {
  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path, recursive = TRUE)
  path <- normalizePath(path)

  trivial_conf = list(
    trivial = list(type = "trivial")
  )
  model_spec$flavors <- c(model_spec$flavors, trivial_conf)
  mlflow:::mlflow_write_model_spec(path, model_spec)
}

mlflow_load_flavor.mlflow_flavor_trivial <- function(flavor, model_path) {
  list(flavor = flavor)
}

library(testthat)
library(mlflow)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  message("Current working directory: ", getwd())
  mlflow_home <- Sys.getenv("MLFLOW_HOME", "../../../../.")
  message('MLFLOW_HOME: ', mlflow_home)
  test_check("mlflow")
}
