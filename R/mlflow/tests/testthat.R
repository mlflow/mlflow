library(testthat)
library(mlflow)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  mlflow_install()
  reticulate::conda_install("r-mlflow", "../../../.", pip = TRUE)

  test_check("mlflow")
}
