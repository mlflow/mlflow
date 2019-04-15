context("Tracking")

test_that("mlflow_create_experiment() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  experiment_1 <- mlflow_create_experiment(client = client, "exp_name", "art_loc")
  experiment_2 <- mlflow_get_experiment(client = client, experiment_1$experiment_id)
  expect_identical(experiment_1$name, "exp_name")
  expect_identical(experiment_1$artifact_location, "art_loc")
  expect_identical(experiment_1, experiment_2)
})

test_that("mlflow_list_experiments() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  mlflow_create_experiment(client = client, "foo1", "art_loc1")
  mlflow_create_experiment(client = client, "foo2", "art_loc2")
  experiments_list <- mlflow_list_experiments(client = client)
  expect_setequal(experiments_list$experiment_id, c("0", "1", "2"))
  expect_setequal(experiments_list$name, c("Default", "foo1", "foo2"))
  default_artifact_loc <- paste(getwd(), "/mlruns/0", sep = "")
  expect_setequal(experiments_list$artifact_location, c(default_artifact_loc,
                                                        "art_loc1",
                                                        "art_loc2"))
})

test_that("mlflow_get_experiment_by_name() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  expect_error(
    mlflow_get_experiment_by_name(client = client, "exp"),
    "Experiment `exp` not found\\."
  )
  experiment_id <- mlflow_create_experiment(client = client, "exp", "art")$experiment_id
  experiment <- mlflow_get_experiment_by_name(client = client, "exp")
  expect_identical(experiment_id, experiment$experiment_id)
  expect_identical(experiment$name, "exp")
  expect_identical(experiment$artifact_location, "art")
})

test_that("mlflow_create_run()/mlflow_get_run() work properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  run <- mlflow_create_run(
    client = client,
    experiment_id = "0",
    user_id = "user1",
    tags = list(foo = "bar", foz = "baz")
  )

  run <- mlflow_get_run(client = client, run$run_uuid)

  expect_identical(run$user_id, "user1")

  expect_true(
    all(purrr::transpose(run$tags[[1]]) %in%
          list(list(key = "foz", value = "baz"), list(key = "foo", value = "bar"))
        )
  )
})

test_that("mlflow_set_teminated() works properly", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()
  killed_time <- mlflow:::current_time()
  client <- mlflow_client()
  run_info <- mlflow_set_terminated(
    client = client, run_id = mlflow_get_active_run()$run_uuid,
    status = "KILLED", end_time = killed_time
  )
  expect_identical(run_info$status, "KILLED")
  expect_identical(run_info$end_time, as.POSIXct(as.double(c(killed_time)) / 1000, origin = "1970-01-01"))
})

test_that("mlflow_create_experiment() works properly", {
  experiment <- mlflow_create_experiment("test")
  expect_gt(as.numeric(experiment$experiment_id), 0)
})

test_that("mlflow_set_tag() should return NULL invisibly", {
  mlflow_clear_test_dir("mlruns")
  value <- mlflow_set_tag("foo", "bar")
  expect_null(value)
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
