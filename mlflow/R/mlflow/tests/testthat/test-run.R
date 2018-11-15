context("Run")

test_that("mlflow can run and save model", {
  mlflow_clear_test_dir("mlruns")

  mlflow_source("examples/train_example.R")

  expect_true(dir.exists("mlruns"))
  expect_true(dir.exists("mlruns/0"))
  expect_true(file.exists("mlruns/0/meta.yaml"))
})
