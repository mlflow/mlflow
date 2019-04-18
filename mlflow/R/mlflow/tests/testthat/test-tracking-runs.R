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

test_that("mlflow_set_teminated() works properly", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()
  killed_time <- mlflow:::current_time()
  client <- mlflow_client()
  run_info <- mlflow_set_terminated(
    client = client, run_id = mlflow_get_active_run_id(),
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

  mlflow_start_run()

  mlflow_log_metric("mse", 24)
  mlflow_log_metric("mse", 25)
  metric_history <- mlflow_get_metric_history("mse")
  expect_identical(metric_history$key, c("mse", "mse"))
  expect_identical(metric_history$value, c(24, 25))

  mlflow_set_tag("tag_key", "tag_value")
  mlflow_log_param("param_key", "param_value")

  run <- mlflow_active_run()
  expect_identical(run$tags[[1]]$key, "tag_key")
  expect_identical(run$tags[[1]]$value, "tag_value")
  expect_identical(run$params[[1]]$key, "param_key")
  expect_identical(run$params[[1]]$value, "param_value")

  mlflow_end_run()
  expect_error(
    mlflow_active_run(),
    "There is no active run\\."
  )
})
