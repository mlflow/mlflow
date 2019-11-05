context("Installing Python MLflow Package")

expected_conda_env_name <- function() {
  paste("r-mlflow", packageVersion("mlflow"), sep = "-")
}

test_that("MLflow installs into a conda environment with the same name as current Mlflow version", {
  conda_env_name <- mlflow:::mlflow_conda_env_name()
  expect_equal(conda_env_name, expected_conda_env_name())
})

test_that("MLflow uses 'python' executable from correct conda environment", {
  expect_true(grepl(expected_conda_env_name(), mlflow:::python_bin()))
})

test_that("MLflow uses 'mlflow' executable from correct conda environment", {
  expect_true(grepl(expected_conda_env_name(), mlflow:::python_mlflow_bin()))
})
