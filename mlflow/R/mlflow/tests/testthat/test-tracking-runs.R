context("Tracking")

test_that("mlflow_start_run()/mlflow_get_run() work properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  run <- mlflow_start_run(
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

test_that("mlflow_end_run() works properly", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()
  killed_time <- mlflow:::current_time()
  client <- mlflow_client()
  run_info <- mlflow_end_run(
    client = client, run_id = mlflow:::mlflow_get_active_run_id(),
    status = "KILLED", end_time = killed_time
  )
  expect_identical(run_info$status, "KILLED")
  expect_identical(run_info$end_time, as.POSIXct(as.double(c(killed_time)) / 1000, origin = "1970-01-01"))
})

test_that("mlflow_set_tag() should return NULL invisibly", {
  mlflow_clear_test_dir("mlruns")
  value <- mlflow_set_tag("foo", "bar")
  expect_null(value)
})

test_that("logging functionality", {
  mlflow_clear_test_dir("mlruns")

  start_time_lower_bound <- Sys.time()
  mlflow_start_run()

  mlflow_log_metric("mse", 24)
  mlflow_log_metric("mse", 25)

  mlflow_set_tag("tag_key", "tag_value")
  mlflow_log_param("param_key", "param_value")

  run <- mlflow_get_run()
  run_id <- run$run_uuid
  expect_identical(run$tags[[1]]$key, "tag_key")
  expect_identical(run$tags[[1]]$value, "tag_value")
  expect_identical(run$params[[1]]$key, "param_key")
  expect_identical(run$params[[1]]$value, "param_value")

  mlflow_end_run()
  end_time_upper_bound <- Sys.time()
  ended_run <- mlflow_get_run(run_id = run_id)
  run_start_time <- ended_run$start_time
  run_end_time <- ended_run$end_time
  expect_true(difftime(run_start_time, start_time_lower_bound) >= 0)
  expect_true(difftime(run_end_time, end_time_upper_bound) <= 0)
  metric_history <- mlflow_get_metric_history("mse", ended_run$run_uuid)
  expect_identical(metric_history$key, c("mse", "mse"))
  expect_identical(metric_history$value, c(24, 25))
  expect_true(all(difftime(metric_history$timestamp, run_start_time) >= 0))
  expect_true(all(difftime(metric_history$timestamp, run_end_time) <= 0))


  expect_error(
    mlflow_get_run(),
    "`run_id` must be specified when there is no active run\\."
  )
})

test_that("mlflow_end_run() behavior", {
  mlflow_clear_test_dir("mlruns")
  expect_error(
    mlflow_end_run(),
    "There is no active run to end\\."
  )

  run <- mlflow_start_run()
  run_id <- mlflow_id(run)
  mlflow_end_run(run_id = run_id)
  expect_error(
    mlflow_get_run(),
    "`run_id` must be specified when there is no active run\\."
  )

  run <- mlflow_start_run()
  run_id <- mlflow_id(run)
  client <- mlflow_client()
  expect_error(
    mlflow_end_run(client = client),
    "`run_id` must be specified when `client` is specified\\."
  )
  mlflow_end_run(client = client, run_id = run_id)
  expect_error(
    mlflow_get_run(),
    "`run_id` must be specified when there is no active run\\."
  )

  mlflow_start_run()
  run <- mlflow_end_run(status = "KILLED")
  expect_identical(
    run$status,
    "KILLED"
  )
})

test_that("with() errors when not passed active run", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  mlflow_set_experiment("exp1")
  run <- mlflow_start_run(client = client)
  expect_error(
    with(run, {
      mlflow_log_metric("mse", 25)
    }),
    # TODO: somehow this isn't matching "`with()` should only be used with `mlflow_start_run()`\\."
    NULL
  )
})

test_that("mlflow_log_batch() works", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()
  mlflow_log_batch(
    metrics = list(mse = 21, rmse = 42),
    params = list(l1 = 0.01, optimizer = "adam"),
    tags = list(model_type = "regression", data_year = "2015")
  )

  run <- mlflow_get_run()
  metrics <- run$metrics[[1]]
  params <- run$params[[1]]
  tags <- run$tags[[1]]

  expect_setequal(
    metrics$key,
    c("mse", "rmse")
  )
  expect_setequal(
    metrics$value,
    c(21, 42)
  )

  expect_setequal(
    params$key,
    c("optimizer", "l1")
  )

  expect_setequal(
    params$value,
    c("adam", "0.01")
  )

  expect_setequal(
    tags$key,
    c("model_type", "data_year")
  )

  expect_setequal(
    tags$value,
    c("regression", "2015")
  )
})

test_that("mlflow_log_batch() works with timestamp", {
  mlflow_clear_test_dir("mlruns")
  my_time <- mlflow:::current_time()

  mlflow_log_batch(
    metrics = list(accuracy = 0.98, accuracy = 0.99),
    timestamps = rep(my_time, 2)
  )

  metric_history <- mlflow_get_metric_history("accuracy")

  expect_equal(
    as.double(metric_history$timestamp),
    rep(my_time, 2) / 1000
  )
})
