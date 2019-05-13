context("Installing Python MLflow Package")

test_that("MLflow installs into a conda environment with the same name as current Mlflow version") {
  conda_env_name <- mlflow:::mlflow_conda_env_name()
  expect_equal(conda_env_name, paste("r-mlflow", packageVersion("mlflow"), sep="-"))
}
