library(testthat)
library(mlflow)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  test_check("mlflow")
}
