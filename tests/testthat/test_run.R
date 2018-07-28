context("Run")

test_that("mlflow can run and save model", {
  mlflow_run("examples/train_save.R")
})
