context("REST wrappers")

test_that("mlflow_create_experiment() works properly", {
  experiment_id <- mlflow_create_experiment("exp_name", "art_loc")
  experiment <- mlflow_get_experiment(experiment_id)
  expect_identical(experiment$experiment$name, "exp_name")
  expect_identical(experiment$experiment$artifact_location, "art_loc")
})

test_that("mlflow_create_experiment() `activate` parameter is respected", {
  experiment_id <- mlflow_create_experiment("foo1", "art_loc")
  expect_identical(mlflow_active_experiment(), experiment_id)
  mlflow_create_experiment("foo2", "art_loc", activate = FALSE)
  expect_identical(mlflow_active_experiment(), experiment_id)
})

test_that("mlflow_create_experiment() checks for existing experiment with same name", {
  expect_message(
    mlflow_create_experiment("foo2", "art_loc"),
    "Experiment with name \"foo2\" already exists\\."
  )
})

test_that("mlflow_list_experiments() works properly", {
  cat(getwd())
  mlflow_clear_test_dir("mlruns")
  mlflow_create_experiment("foo1", "art_loc1")
  mlflow_create_experiment("foo2", "art_loc2")
  experiments_list <- mlflow_list_experiments()
  expect_identical(experiments_list$experiment_id, c("0", "1", "2"))
  expect_identical(experiments_list$name, c("Default", "foo1", "foo2"))
  expect_identical(experiments_list$artifact_location, c("mlruns/0", "art_loc1", "art_loc2"))
})
