context("Tracking - Experiments")

test_that("mlflow_create/get_experiment() basic functionality (fluent)", {
  mlflow_clear_test_dir("mlruns")

  experiment_1 <- mlflow_create_experiment("exp_name", "art_loc")
  experiment_1_id <- mlflow_id(experiment_1)
  experiment_1a <- mlflow_get_experiment(experiment_id = experiment_1_id)
  experiment_1b <- mlflow_get_experiment(name = "exp_name")

  expect_identical(experiment_1a, experiment_1b)
  expect_identical(experiment_1$artifact_location, "art_loc")
})

test_that("mlflow_create/get_experiment() basic functionality (client)", {
  mlflow_clear_test_dir("mlruns")

  client <- mlflow_client()

  experiment_1 <- mlflow_create_experiment(client = client, "exp_name", "art_loc")
  experiment_1_id <- mlflow_id(experiment_1)
  experiment_1a <- mlflow_get_experiment(client = client, experiment_id = experiment_1_id)
  experiment_1b <- mlflow_get_experiment(client = client, name = "exp_name")

  expect_identical(experiment_1a, experiment_1b)
  expect_identical(experiment_1$artifact_location, "art_loc")
})

test_that("mlflow_get_experiment() not found error", {
  mlflow_clear_test_dir("mlruns")

  expect_error(
    mlflow_get_experiment(experiment_id = "42"),
    "Could not find experiment with ID 42"
    )
})

test_that("mlflow_list_experiments() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  mlflow_create_experiment(client = client, "foo1", "art_loc1")
  mlflow_create_experiment(client = client, "foo2", "art_loc2")

  # client
  experiments_list <- mlflow_list_experiments(client = client)
  expect_setequal(experiments_list$experiment_id, c("0", "1", "2"))
  expect_setequal(experiments_list$name, c("Default", "foo1", "foo2"))
  default_artifact_loc <- paste(getwd(), "/mlruns/0", sep = "")
  expect_setequal(experiments_list$artifact_location, c(default_artifact_loc,
                                                        "art_loc1",
                                                        "art_loc2"))

  # fluent
  experiments_list <- mlflow_list_experiments()
  expect_setequal(experiments_list$experiment_id, c("0", "1", "2"))
  expect_setequal(experiments_list$name, c("Default", "foo1", "foo2"))
  default_artifact_loc <- paste(getwd(), "/mlruns/0", sep = "")
  expect_setequal(experiments_list$artifact_location, c(default_artifact_loc,
                                                        "art_loc1",
                                                        "art_loc2"))

  # Returns NULL when no experiments found
  expect_null(mlflow_list_experiments("DELETED_ONLY"))

  # `view_type` is respected
  mlflow_delete_experiment(experiment_id = "1")
  deleted_experiments <- mlflow_list_experiments("DELETED_ONLY")
  expect_identical(deleted_experiments$name, "foo1")
})


test_that("mlflow_get_experiment_by_name() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  expect_error(
    mlflow_get_experiment(client = client, "exp"),
    "Experiment `exp` not found\\."
  )
  experiment_id <- mlflow_create_experiment(client = client, "exp", "art") %>%
    mlflow_id()
  experiment <- mlflow_get_experiment(client = client, name = "exp")
  expect_identical(experiment_id, experiment$experiment_id)
  expect_identical(experiment$name, "exp")
  expect_identical(experiment$artifact_location, "art")
})

test_that("infer experiment id works properly", {
  mlflow_clear_test_dir("mlruns")
  experiment_id <- mlflow_create_experiment("test")$experiment_id
  Sys.setenv(MLFLOW_EXPERIMENT_NAME = "test")
  expect_true(experiment_id == mlflow_infer_experiment_id())
  Sys.unsetenv("MLFLOW_EXPERIMENT_NAME")
  Sys.setenv(MLFLOW_EXPERIMENT_ID = experiment_id)
  expect_true(experiment_id == mlflow_infer_experiment_id())
  Sys.unsetenv("MLFLOW_EXPERIMENT_ID")
  mlflow_set_experiment("test")
  expect_true(experiment_id == mlflow_infer_experiment_id())
})

test_that("experiment setting works", {
  mlflow_clear_test_dir("mlruns")
  exp1 <- mlflow_create_experiment("exp1")
  exp2 <- mlflow_create_experiment("exp2")
  exp1_set <- mlflow_set_experiment(experiment_name = "exp1")
  expect_identical(exp1, exp1_set)
  expect_identical(exp1, mlflow_get_experiment())
  exp2_set <- mlflow_set_experiment(experiment_id = mlflow_id(exp2))
  expect_identical(exp2, exp2_set)
  expect_identical(exp2, mlflow_get_experiment())
})

test_that("mlflow_set_experiment() creates experiments", {
  mlflow_clear_test_dir("mlruns")
  mlflow_set_experiment(experiment_name = "foo", artifact_location = "artifact/location")
  experiment <- mlflow_get_experiment()
  expect_identical(experiment$artifact_location, "artifact/location")
  expect_identical(experiment$name, "foo")
})
