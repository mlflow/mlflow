context("Tracking")

test_that("mlflow_start_run()/mlflow_get_run() work properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  run <- mlflow_start_run(
    client = client,
    experiment_id = "0",
    tags = list(foo = "bar", foz = "baz", mlflow.user = "user1")
  )

  run <- mlflow_get_run(client = client, run$run_uuid)

  expect_identical(run$user_id, "user1")

  expect_true(
    all(purrr::transpose(run$tags[[1]]) %in%
      list(
        list(key = "foz", value = "baz"),
        list(key = "foo", value = "bar"),
        list(key = "mlflow.user", value = "user1")
      )
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
  tags <- run$tags[[1]]
  expect_identical("tag_value", tags$value[tags$key == "tag_key"])
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
  expect_identical(metric_history$step, c(0, 0))
  expect_true(all(difftime(metric_history$timestamp, run_start_time) >= 0))
  expect_true(all(difftime(metric_history$timestamp, run_end_time) <= 0))

  expect_error(
    mlflow_get_run(),
    "`run_id` must be specified when there is no active run\\."
  )
})

test_that("mlflow_log_metric() rounds step and timestamp inputs", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()

  step_inputs <- runif(10, min = -20, max = 100)
  for (step_input in step_inputs) {
    mlflow_log_metric(key = "step_metric",
                      value = runif(1),
                      step = step_input,
                      timestamp = 100)
  }
  expect_setequal(
    mlflow_get_metric_history("step_metric")$step,
    round(step_inputs)
  )

  timestamp_inputs <- runif(10, 1000, 100000)
  for (timestamp_input in timestamp_inputs) {
    mlflow_log_metric(key = "timestamp_metric",
                      value = runif(1),
                      step = 0,
                      timestamp = timestamp_input)
  }
  expect_setequal(
    mlflow_get_metric_history("timestamp_metric")$timestamp,
    purrr::map(round(timestamp_inputs), mlflow:::milliseconds_to_date)
  )
})

test_that("mlflow_log_metric() with step produces expected metric data", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()

  metric_key_1 <- "test_metric_1"
  mlflow_log_metric(key = metric_key_1,
                    value = 1.2,
                    step = -2,
                    timestamp = 300)
  mlflow_log_metric(key = metric_key_1,
                    value = 137.4,
                    timestamp = 100)
  mlflow_log_metric(key = metric_key_1,
                    value = -20,
                    timestamp = 200)

  metric_key_2 <- "test_metric_2"
  mlflow_log_metric(key = metric_key_2,
                    value = 14,
                    step = 120)
  mlflow_log_metric(key = metric_key_2,
                    value = 56,
                    step = 137)
  mlflow_log_metric(key = metric_key_2,
                    value = 20,
                    step = -5)

  run <- mlflow_get_run()
  metrics <- run$metrics[[1]]
  expect_setequal(
    metrics$key,
    c("test_metric_1", "test_metric_2")
  )
  expect_setequal(
    metrics$value,
    c(-20, 56)
  )
  expect_setequal(
    metrics$step,
    c(0, 137)
  )

  metric_history_1 <- mlflow_get_metric_history("test_metric_1")
  expect_setequal(
    metric_history_1$value,
    c(1.2, 137.4, -20)
  )
  expect_setequal(
    metric_history_1$timestamp,
    purrr::map(c(300, 100, 200), mlflow:::milliseconds_to_date)
  )
  expect_setequal(
    metric_history_1$step,
    c(-2, 0, 0)
  )

  metric_history_2 <- mlflow_get_metric_history("test_metric_2")
  expect_setequal(
    metric_history_2$value,
    c(14, 56, 20)
  )
  expect_setequal(
    metric_history_2$step,
    c(120, 137, -5)
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
    metrics = data.frame(key = c("mse", "mse", "rmse", "rmse"),
                         value = c(21, 23, 42, 36),
                         timestamp = c(100, 200, 300, 300),
                         step = c(-4, 1, 7, 3)),
    params = data.frame(key = c("l1", "optimizer"), value = c(0.01, "adam")),
    tags = data.frame(key = c("model_type", "data_year"),
                      value = c("regression", "2015"))
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
    c(23, 42)
  )
  expect_setequal(
    metrics$timestamp,
    purrr::map(c(200, 300), mlflow:::milliseconds_to_date)
  )
  expect_setequal(
    metrics$step,
    c(1, 7)
  )

  metric_history <- mlflow_get_metric_history("mse")
  expect_setequal(
    metric_history$value,
    c(21, 23)
  )
  expect_setequal(
    metric_history$timestamp,
    purrr::map(c(100, 200), mlflow:::milliseconds_to_date)
  )
  expect_setequal(
    metric_history$step,
    c(-4, 1)
  )

  expect_setequal(
    params$key,
    c("optimizer", "l1")
  )

  expect_setequal(
    params$value,
    c("adam", "0.01")
  )

  expect_identical("regression", tags$value[tags$key == "model_type"])
  expect_identical("2015", tags$value[tags$key == "data_year"])
})

test_that("mlflow_log_batch() throws for mismatched input data columns", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()
  error_text_regexp <- "*input dataframe must contain exactly the following columns*"

  expect_error(
    mlflow_log_batch(
      metrics = data.frame(key = c("mse"), value = c(10))
    ),
    regexp = error_text_regexp
  )
  expect_error(
    mlflow_log_batch(
      metrics = data.frame(bad_column = c("bad"))
    ),
    regexp = error_text_regexp
  )
  expect_error(
    mlflow_log_batch(
      metrics = data.frame(
       key = c("mse"),
       value = c(10),
       timestamp = c(100),
       step = c(1),
       bad_column = c("bad")
      )
    ),
    regexp = error_text_regexp
  )

  expect_error(
    mlflow_log_batch(
      params = data.frame(key = c("alpha"))
    ),
    regexp = error_text_regexp
  )
  expect_error(
    mlflow_log_batch(
      params = data.frame(bad_column = c("bad"))
    ),
    regexp = error_text_regexp
  )
  expect_error(
    mlflow_log_batch(
      params = data.frame(
       key = c("alpha"),
       value = c(10),
       bad_column = c("bad")
      )
    ),
    regexp = error_text_regexp
  )

  expect_error(
    mlflow_log_batch(
      tags = data.frame(key = c("my_tag"))
    ),
    regexp = error_text_regexp
  )
  expect_error(
    mlflow_log_batch(
      tags = data.frame(bad_column = c("bad"))
    ),
    regexp = error_text_regexp
  )
  expect_error(
    mlflow_log_batch(
      tags = data.frame(
       key = c("my_tag"),
       value = c("some tag info"),
       bad_column = c("bad")
      )
    ),
    regexp = error_text_regexp
  )

  expect_error(
    mlflow_log_batch(
      metrics = data.frame(
       bad_column = c("bad1")
      ),
      params = data.frame(
       another_bad_column = c("bad2")
      ),
      tags = data.frame(
       one_more_bad_column = c("bad3")
      )
    ),
    regexp = error_text_regexp
  )
})

test_that("mlflow_log_batch() throws for missing entries", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()
  error_text_regexp <- "*input dataframe contains a missing \\('NA'\\) entry*"

  expect_error(
    mlflow_log_batch(
      metrics = data.frame(key = c("mse", "rmse", "na_metric"),
                           value = c(21, 42, NA),
                           timestamp = c(100, 200, 300),
                           step = c(-4, 1, 3))
    ),
    regexp = error_text_regexp
  )

  expect_error(
    mlflow_log_batch(
      metrics = data.frame(key = c("mse", "rmse", "na_metric"),
                           value = c(21, 42, 9.2),
                           timestamp = c(NA, 200, 300),
                           step = c(-4, 1, NA))
    ),
    regexp = error_text_regexp
  )

  expect_error(
    mlflow_log_batch(
      params = data.frame(key = c(NA, "alpha"),
                          value = c("0.5", "2"))
    ),
    regexp = error_text_regexp
  )
  expect_error(
    mlflow_log_batch(
      params = data.frame(key = c("n_layers", "alpha"),
                          value = c("4", NA))
    ),
    regexp = error_text_regexp
  )

  expect_error(
    mlflow_log_batch(
      tags = data.frame(key = c("first_tag", NA),
                        value = c("some tag info", "more tag info"))
    ),
    regexp = error_text_regexp
  )
  expect_error(
    mlflow_log_batch(
      tags = data.frame(key = c("first_tag", "second_tag"),
                        value = c(NA, NA))
    ),
    regexp = error_text_regexp
  )

  expect_error(
    mlflow_log_batch(
      metrics = data.frame(key = c("mse", "rmse", "na_metric"),
                           value = c(21, 42, 37),
                           timestamp = c(100, 200, 300),
                           step = c(-4, NA, 3)),
      params = data.frame(key = c("l1", "optimizer"), value = c(NA, "adam")),
      tags = data.frame(key = c(NA, "data_year"),
                        value = c("regression", NA))
    ),
    regexp = error_text_regexp
  )
})
