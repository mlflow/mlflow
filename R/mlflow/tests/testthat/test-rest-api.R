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
