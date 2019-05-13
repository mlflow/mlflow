library(testthat)
library(mlflow)


if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  conda_env_name <- mlflow:::mlflow_conda_env_name()
  if (!conda_env_name %in% reticulate::conda_list()$name) {
    mlflow_install()
    message("Current working directory: ", getwd())
    mlflow_home <- Sys.getenv("MLFLOW_HOME", "../../../../.")
    reticulate::conda_install(conda_env_name, mlflow_home, pip = TRUE)
  }
  test_check("mlflow")
}
