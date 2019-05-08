library(testthat)
library(mlflow)


if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  if (!"r-mlflow" %in% reticulate::conda_list()$name) {
    mlflow_install()
    message("Current working directory: ", getwd())
    mlflow_home <- Sys.getenv("MLFLOW_HOME", "../../../../.")
    reticulate::conda_install("r-mlflow", mlflow_home, pip = TRUE)
  }
  test_check("mlflow")
}
