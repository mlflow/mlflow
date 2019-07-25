context("Bypass conda")


test_that("MLflow finds MLFLOW_PYTHON_BIN environment variable", {
  orig <- Sys.getenv("MLFLOW_PYTHON_BIN")
  expected_path <- tempfile("python")
  file.create(expected_path)
  Sys.setenv(MLFLOW_PYTHON_BIN = expected_path)
  python_bin <- mlflow:::get_python_bin()
  expect_equal(python_bin, expected_path)
  # Clean up
  file.remove(expected_path)
  Sys.setenv(MLFLOW_PYTHON_BIN = orig)
})

test_that("MLflow finds MLFLOW_BIN environment variable", {
  orig_global <- mlflow:::.globals$python_bin
  rm("python_bin", envir = mlflow:::.globals)
  orig_env <- Sys.getenv("MLFLOW_BIN")
  expected_path <- tempfile("mlflow")
  file.create(expected_path)
  Sys.setenv(MLFLOW_BIN = expected_path)
  mlflow_bin <- mlflow:::python_mlflow_bin()
  expect_equal(mlflow_bin, expected_path)
  # Clean up
  file.remove(expected_path)
  Sys.setenv(MLFLOW_BIN = orig_env)
  assign("python_bin", orig_global, envir = mlflow:::.globals)
})

test_that("MLflow defaults MLFLOW_BIN from the same directory than MLFLOW_PYTHON_BIN", {
  orig <- Sys.getenv("MLFLOW_PYTHON_BIN")
  expected_python_path <- tempfile("python")
  expected_mlflow_path <- paste(tempdir(), "mlflow", sep = "/")
  file.create(expected_python_path)
  Sys.setenv(MLFLOW_PYTHON_BIN = expected_python_path)
  expected_fail <- try(mlflow:::python_mlflow_bin(), silent = TRUE)
  expect_equal(class(expected_fail), "try-error")
  file.create(expected_mlflow_path)
  mlflow_bin <- mlflow:::python_mlflow_bin()
  expect_equal(mlflow_bin, expected_mlflow_path)
  # Clean up
  file.remove(expected_python_path)
  file.remove(expected_mlflow_path)
  Sys.setenv(MLFLOW_PYTHON_BIN = orig)
})

test_that("MLflow bypasses `python_conda_home()` when Python and MLflow executables are provided", {
  orig <- Sys.getenv("MLFLOW_PYTHON_BIN")
  expected_python_path <- tempfile("python")
  expected_mlflow_path <- paste(tempdir(), "mlflow", sep = "/")
  file.create(expected_python_path)
  Sys.setenv(MLFLOW_PYTHON_BIN = expected_python_path)
  file.create(expected_mlflow_path)
  conda_home <- mlflow:::python_conda_home()
  expect_equal(conda_home, "")
  # Clean up
  file.remove(expected_python_path)
  file.remove(expected_mlflow_path)
  Sys.setenv(MLFLOW_PYTHON_BIN = orig)
})
