library(testthat)
library(mlflow)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  message("Current working directory: ", getwd())
  mlflow_home <- Sys.getenv("MLFLOW_HOME", "../../../.")
  message('MLFLOW_HOME: ', mlflow_home)
  test_check("mlflow")
}
