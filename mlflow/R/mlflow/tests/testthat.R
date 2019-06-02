library(testthat)
library(mlflow)
library(reticulate)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  mlflow:::mlflow_maybe_create_conda_env()
  message("Current working directory: ", getwd())
  mlflow_home <- Sys.getenv("MLFLOW_HOME", "../../../../.")
  conda_install(c(mlflow_home), envname = mlflow:::mlflow_conda_env_name(), pip = TRUE)
  test_check("mlflow")
}
