context("Run")

test_that("mlflow can run and save model", {
  mlflow_run("examples/train_save.R")
})

test_that("mlflow can run and save model with context", {
  mlflow_run("examples/train_save.R")
})
