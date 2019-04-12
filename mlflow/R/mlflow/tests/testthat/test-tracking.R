context("Tracking")

test_that("mlflow_client_create_experiment() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  experiment_id <- mlflow_client_create_experiment(client = client, "exp_name", "art_loc")
  experiment <- mlflow_client_get_experiment(client = client, experiment_id)
  expect_identical(experiment$experiment$name, "exp_name")
  expect_identical(experiment$experiment$artifact_location, "art_loc")
})

test_that("mlflow_client_list_experiments() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  mlflow_client_create_experiment(client = client, "foo1", "art_loc1")
  mlflow_client_create_experiment(client = client, "foo2", "art_loc2")
  experiments_list <- mlflow_client_list_experiments(client = client)
  expect_setequal(experiments_list$experiment_id, c("0", "1", "2"))
  expect_setequal(experiments_list$name, c("Default", "foo1", "foo2"))
  default_artifact_loc <- paste(getwd(), "/mlruns/0", sep = "")
  expect_setequal(experiments_list$artifact_location, c(default_artifact_loc,
                                                        "art_loc1",
                                                        "art_loc2"))
})

test_that("mlflow_client_get_experiment() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  experiment_id <- mlflow_client_create_experiment(client = client, "foo1", "art_loc1")
  experiment <- mlflow_client_get_experiment(client = client, experiment_id)
  expect_identical(experiment$experiment$experiment_id, experiment_id)
  expect_identical(experiment$experiment$name, "foo1")
  expect_identical(experiment$experiment$artifact_location, "art_loc1")
})


test_that("mlflow_client_get_experiment_by_name() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  experiment <- mlflow_client_get_experiment_by_name(client = client, "exp")
  expect_null(experiment)
  experiment_id <- mlflow_client_create_experiment(client = client, "exp", "art")
  experiment <- mlflow_client_get_experiment_by_name(client = client, "exp")
  expect_identical(experiment_id, experiment$experiment_id)
  expect_identical(experiment$name, "exp")
  expect_identical(experiment$artifact_location, "art")
})

test_that("mlflow_client_create_run()/mlflow_client_get_run() work properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  create_run_response <- mlflow_client_create_run(
    client = client,
    experiment_id = "0",
    user_id = "user1",
    run_name = "run1",
    tags = list(foo = "bar", foz = "baz")
  )

  run <- mlflow_client_get_run(client = client, create_run_response$info$run_uuid)
  run_info <- run$info

  expect_identical(run_info$user_id, "user1")

  actual_tags <- run$data$tags %>%
    unname() %>%
    purrr::transpose() %>%
    purrr::map(purrr::flatten_chr)
  expected_tags <- list(c("foo", "bar"), c("foz", "baz"))

  expect_true(all(expected_tags %in% actual_tags))
})

test_that("mlflow_client_set_teminated() works properly", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()
  killed_time <- mlflow:::current_time()
  client <- mlflow_client()
  run_info <- mlflow_client_set_terminated(
    client = client, run_id = mlflow_active_run()$info$run_uuid,
    status = "KILLED", end_time = killed_time
  )
  expect_identical(run_info$status, "KILLED")
  expect_identical(run_info$end_time, as.POSIXct(as.double(c(killed_time)) / 1000, origin = "1970-01-01"))
})

test_that("mlflow_create_experiment() works properly", {
  experiment <- mlflow_create_experiment("test")
  expect_gt(as.numeric(experiment), 0)
})

test_that("mlflow_set_tag() should return NULL invisibly", {
  mlflow_clear_test_dir("mlruns")
  value <- mlflow_set_tag("foo", "bar")
  expect_null(value)
})

test_that("infer experiment id works properly", {
  mlflow_clear_test_dir("mlruns")
  experiment_id <- mlflow_create_experiment("test")
  Sys.setenv(MLFLOW_EXPERIMENT_NAME = "test")
  expect_true(experiment_id == mlflow_infer_experiment_id())
  Sys.unsetenv("MLFLOW_EXPERIMENT_NAME")
  Sys.setenv(MLFLOW_EXPERIMENT_ID = experiment_id)
  expect_true(experiment_id == mlflow_infer_experiment_id())
  Sys.unsetenv("MLFLOW_EXPERIMENT_ID")
  mlflow_set_experiment("test")
  expect_true(experiment_id == mlflow_infer_experiment_id())
})
