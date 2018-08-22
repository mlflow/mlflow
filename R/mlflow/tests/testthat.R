library(testthat)
library(mlflow)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  mlflow_install()

  mlflow_home <- Sys.getenv("MLFLOW_HOME", "../../../.")
  reticulate::conda_install("r-mlflow", mlflow_home, pip = TRUE)

  test_check("mlflow")
}
