context("Run")

teardown({
  mlflow_clear_test_dir("mlruns")
})

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


test_that("mlflow_run passes all numbers as non-scientific", {
  # we can only be sure conversion is actively avoided
  # if default formatting turns into scientific.
  expect_equal(as.character(10e4), "1e+05")
  with_mock(.env = "mlflow", mlflow_cli = function(...){
    args <- c(...)
    expect_equal(sum("scientific=100000" == args), 1)
    expect_equal(sum("non_scientific=30000" == args), 1)
    list(stderr = "=== Run (ID '48734e7e2e8f44228a11c0c2cbcdc8b0') succeeded ===")
  }, {
    mlflow_run("project", parameters = c(scientific = 10e4, non_scientific = 30000))
  })
})

test_that("active experiment is set when starting a run with experiment specified", {
  mlflow_clear_test_dir("mlruns")
  id <- mlflow_create_experiment("one-more")
  mlflow_start_run(experiment_id = id)
  expect_equal(
    mlflow_get_experiment()$experiment_id,
    id
  )
  mlflow_end_run()
})

test_that("active experiment is set when starting a run without experiment specified", {
  mlflow_clear_test_dir("mlruns")
  id <- mlflow_create_experiment("second-exp")
  mlflow_start_run()
  expect_equal(
    mlflow_get_experiment()$experiment_id,
    "0"
  )
  mlflow_end_run()
})
