library(testthat)
library(mlflow)

test_that <- function(x, y) {
  write(paste("STARTING testing", x, "\n", sep = " "), stderr())
  testthat::test_that(x,y)
  write(paste("DONE testing", x, "\n", sep = " "), stderr())
  write(paste("tracking uri", mlflow_get_tracking_uri(), "\n", sep = " "), stderr())
}

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  if (!"r-mlflow" %in% reticulate::conda_list()$name) {
    mlflow_install()
    message("Current working directory: ", getwd())
    mlflow_home <- Sys.getenv("MLFLOW_HOME", "../../../../.")
    reticulate::conda_install("r-mlflow", mlflow_home, pip = TRUE)
  }
  test_check("mlflow")
}
