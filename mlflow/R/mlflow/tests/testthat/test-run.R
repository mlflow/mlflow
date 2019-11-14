context("Run")

test_that("mlflow can run and save model", {
  mlflow_clear_test_dir("mlruns")

  mlflow_source("examples/train_example.R")

  expect_true(dir.exists("mlruns"))
  expect_true(dir.exists("mlruns/0"))
  expect_true(file.exists("mlruns/0/meta.yaml"))
})


test_that("mlflow run uses active experiment if not specified", {
  with_mock(.env = "mlflow", mlflow_get_active_experiment_id = function() {
    123}, {
      with_mock(.env = "mlflow", mlflow_cli = function(...){
        args <- list(...)
        expect_true("--experiment-id" %in% args)
        expect_false("--experiment-name" %in% args)
        id <- which(args == "--experiment-id") + 1
        expect_true(args[[id]] == 123)
        list(stderr = "=== Run (ID '48734e7e2e8f44228a11c0c2cbcdc8b0') succeeded ===")
      }, {
        mlflow_run("project")
      })
      with_mock(.env = "mlflow", mlflow_cli = function(...){
        args <- list(...)
        expect_true("--experiment-id" %in% args)
        expect_false("--experiment-name" %in% args)
        id <- which(args == "--experiment-id") + 1
        expect_true(args[[id]] == 321)
        list(stderr = "=== Run (ID '48734e7e2e8f44228a11c0c2cbcdc8b0') succeeded ===")
      }, {
        mlflow_run("project", experiment_id = 321)
      })
      with_mock(.env = "mlflow", mlflow_cli = function(...){
        args <- list(...)
        expect_false("--experiment-id" %in% args)
        expect_true("--experiment-name" %in% args)
        id <- which(args == "--experiment-name") + 1
        expect_true(args[[id]] == "name")
        list(stderr = "=== Run (ID '48734e7e2e8f44228a11c0c2cbcdc8b0') succeeded ===")
      }, {
        mlflow_run("project", experiment_name = "name")
      })
  })
})
