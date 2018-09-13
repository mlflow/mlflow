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
  mlflow_clear_test_dir("mlruns")
  mlflow_create_experiment("foo1", "art_loc1")
  mlflow_create_experiment("foo2", "art_loc2")
  experiments_list <- mlflow_list_experiments()
  expect_setequal(experiments_list$experiment_id, c("0", "1", "2"))
  expect_setequal(experiments_list$name, c("Default", "foo1", "foo2"))
  expect_setequal(experiments_list$artifact_location, c("mlruns/0", "art_loc1", "art_loc2"))
})

test_that("mlflow_get_experiment() works properly", {
  mlflow_clear_test_dir("mlruns")
  experiment_id <- mlflow_create_experiment("foo1", "art_loc1")
  experiment <- mlflow_get_experiment(experiment_id)
  expect_identical(experiment$experiment$experiment_id, experiment_id)
  expect_identical(experiment$experiment$name, "foo1")
  expect_identical(experiment$experiment$artifact_location, "art_loc1")
})

test_that("mlflow_create_run()/mlflow_get_run() work properly", {
  create_run_response <- mlflow_create_run(
    user_id = "user1",
    run_name = "run1",
    tags = list(foo = "bar", foz = "baz")
  )

  run <- mlflow_get_run(create_run_response$run_uuid)
  run_info <- run$info

  expect_identical(run_info$user_id, "user1")

  actual_tags <- run$data$tags %>%
    unname() %>%
    purrr::transpose() %>%
    purrr::map(purrr::flatten_chr)
  expected_tags <- list(c("foo", "bar"), c("foz", "baz"))

  expect_true(all(expected_tags %in% actual_tags))
})

test_that("mlflow_log_param()/mlflow_get_param() work properly", {
  mlflow_disconnect()
  some_param <- mlflow_log_param("some_param", 42)
  expect_identical(some_param, 42)
  expect_identical(
    mlflow_get_param("some_param"),
    data.frame(key = "some_param", value = "42", stringsAsFactors = FALSE)
  )
})

test_that("mlflow_get_param() requires `run_uuid` when there is no active run", {
  mlflow_disconnect()
  expect_error(
    mlflow_get_param("some_param"),
    "`run_uuid` must be specified when there is no active run\\."
  )
})

test_that("mlflow_log_metric()/mlflow_get_metric() work properly", {
  mlflow_disconnect()
  log_time <- mlflow:::current_time()
  some_metric <- mlflow_log_metric("some_metric", 42, timestamp = log_time)
  expect_identical(some_metric, 42)
  expect_identical(
    mlflow_get_metric("some_metric"),
    data.frame(
      key = "some_metric", value = 42,
      timestamp = as.POSIXct(as.double(log_time) / 1000, origin = "1970-01-01"),
      stringsAsFactors = FALSE
    )
  )
})

test_that("mlflow_get_metric() requires `run_uuid` when there is no active run", {
  mlflow_disconnect()
  expect_error(
    mlflow_get_metric("some_metric"),
    "`run_uuid` must be specified when there is no active run\\."
  )
})

test_that("mlflow_get_metric_history() works properly", {
  mlflow_disconnect()
  log_time1 <- mlflow:::current_time()
  some_metric1 <- mlflow_log_metric("some_metric", 42, timestamp = log_time1)
  log_time2 <- mlflow:::current_time()
  some_metric2 <- mlflow_log_metric("some_metric", 91, timestamp = log_time2)
  metric_history <- mlflow_get_metric_history("some_metric")
  expected <- data.frame(
    key = c("some_metric", "some_metric"),
    value = c(some_metric1, some_metric2),
    timestamp = as.POSIXct(as.double(c(log_time1, log_time2)) / 1000, origin = "1970-01-01"),
    stringsAsFactors = FALSE
  )
  expect_identical(metric_history, expected)
})

test_that("mlflow_update_run() works properly", {
  mlflow_disconnect()
  mlflow_start_run()
  killed_time <- mlflow:::current_time()
  run_info <- mlflow_update_run(status = "KILLED", end_time = killed_time)
  expect_identical(run_info$status, "KILLED")
  expect_identical(run_info$end_time, as.POSIXct(as.double(c(killed_time)) / 1000, origin = "1970-01-01"))
})
