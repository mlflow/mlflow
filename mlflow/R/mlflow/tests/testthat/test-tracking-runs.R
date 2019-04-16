context("Tracking")

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

test_that("mlflow_set_tag() should return NULL invisibly", {
  mlflow_clear_test_dir("mlruns")
  value <- mlflow_set_tag("foo", "bar")
  expect_null(value)
})
