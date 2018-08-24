context("REST wrappers")

test_that("mlflow_create_experiment works properly", {
  experiment_id <- mlflow_create_experiment("exp_name", "art_loc")
  experiment <- mlflow_get_experiment(experiment_id)
  expect_identical(experiment$experiment$name, "exp_name")
  expect_identical(experiment$experiment$artifact_location, "art_loc")
})
