library(testthat)
library(mlflow)
library(reticulate)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  mlflow:::mlflow_maybe_create_conda_env(python_version = "3.6")
  message("Current working directory: ", getwd())
  mlflow_home <- Sys.getenv("MLFLOW_HOME", "../../../../.")
  message('MLFLOW_HOME: ', mlflow_home)
  conda_install(c(mlflow_home), envname = mlflow:::mlflow_conda_env_name(), pip = TRUE)
  test_check("mlflow")
}
