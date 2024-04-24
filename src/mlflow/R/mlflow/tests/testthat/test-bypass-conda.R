context("Bypass conda")


test_that("MLflow finds MLFLOW_PYTHON_BIN environment variable", {
  orig_global <- mlflow:::.globals$python_bin
  rm("python_bin", envir = mlflow:::.globals)
  orig <- Sys.getenv("MLFLOW_PYTHON_BIN")
  expected_path <- "/test/python"
  Sys.setenv(MLFLOW_PYTHON_BIN = expected_path)
  python_bin <- mlflow:::get_python_bin()
  expect_equal(python_bin, expected_path)
  # Clean up
  Sys.setenv(MLFLOW_PYTHON_BIN = orig)
  assign("python_bin", orig_global, envir = mlflow:::.globals)
})

test_that("MLflow finds MLFLOW_BIN environment variable", {
  orig_global <- mlflow:::.globals$python_bin
  rm("python_bin", envir = mlflow:::.globals)
  orig_env <- Sys.getenv("MLFLOW_BIN")
  expected_path <- "/test/mlflow"
  Sys.setenv(MLFLOW_BIN = expected_path)
  mlflow_bin <- mlflow:::python_mlflow_bin()
  expect_equal(mlflow_bin, expected_path)
  # Clean up
  Sys.setenv(MLFLOW_BIN = orig_env)
  assign("python_bin", orig_global, envir = mlflow:::.globals)
})
