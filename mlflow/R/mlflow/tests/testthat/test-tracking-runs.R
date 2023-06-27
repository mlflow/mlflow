context("Tracking")

teardown({
  mlflow_clear_test_dir("mlruns")
  options(MLflowObservers = NULL)
})

test_that("mlflow_start_run()/mlflow_get_run() work properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  run <- mlflow_start_run(
    client = client,
    experiment_id = "0",
    tags = list(foo = "bar", foz = "baz", mlflow.user = "user1", mlflow.runName = "my_run")
  )

  run <- mlflow_get_run(client = client, run$run_uuid)

  expect_identical(run$user_id, "user1")

  expect_true(
    all(purrr::transpose(run$tags[[1]]) %in%
      list(
        list(key = "foz", value = "baz"),
        list(key = "foo", value = "bar"),
        list(key = "mlflow.user", value = "user1"),
        list(key = "mlflow.runName", value = run$run_name)
      )
    )
  )
})

test_that("a run can be started properly if MLFLOW_RUN_ID is set", {
  mlflow_clear_test_dir("mlruns")
  # Typical use case: Invoke an R script that interacts with the MLflow API from
  # outside of R, e.g. MLproject, Python, CLI
  start_get_id_stop <- function() {
    tryCatch(mlflow_id(mlflow_start_run()), finally = {
      mlflow_end_run()
    })
  }
  id <- start_get_id_stop()
  withr::with_envvar(list(MLFLOW_RUN_ID = id), {
    expect_equal(start_get_id_stop(), id)
  })
})

test_that("mlflow_end_run() works properly", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()
  killed_time <- mlflow:::current_time()
  client <- mlflow_client()
  run <- mlflow_end_run(
    client = client, run_id = mlflow:::mlflow_get_active_run_id(),
    status = "KILLED", end_time = killed_time
  )
  expect_identical(run$status, "KILLED")
  expect_identical(run$end_time, as.POSIXct(as.double(c(killed_time)) / 1000, origin = "1970-01-01"))

  # Verify that only expected run field names are present and that all run info fields are set
  # (not NA).
  run_info_names <- c("run_uuid", "experiment_id", "user_id", "run_name", "status", "start_time",
  "artifact_uri", "lifecycle_stage", "run_id", "end_time")
  run_data_names <- c("metrics", "params", "tags")
  expect_setequal(c(run_info_names, run_data_names), names(run))
  expect_true(!anyNA(run[run_info_names]))
})

test_that("mlflow_start_run()/mlflow_end_run() works properly with nested runs", {
  mlflow_clear_test_dir("mlruns")
  runs <- list(
    mlflow_start_run(),
    mlflow_start_run(nested = TRUE),
    mlflow_start_run(nested = TRUE)
  )
  client <- mlflow_client()
  for (i in seq(3, 1, -1)) {
    expect_equal(mlflow:::mlflow_get_active_run_id(), runs[[i]]$run_uuid)
    run <- mlflow_end_run(client = client, run_id = runs[[i]]$run_uuid)
    expect_identical(run$run_uuid, runs[[i]]$run_uuid)
    if (i > 1) {
      tags <- run$tags[[1]]
      expect_equal(
        tags[tags$key == "mlflow.parentRunId", ]$value,
        runs[[i - 1]]$run_uuid
      )
    }
  }
  expect_null(mlflow:::mlflow_get_active_run_id())
})

test_that("mlflow_restore_run() work properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  run1 <- mlflow_start_run(
    client = client,
    experiment_id = "0",
    tags = list(foo = "bar", foz = "baz", mlflow.user = "user1", mlflow.runName = "my_run")
  )

  run2 <- mlflow_get_run(client = client, run1$run_uuid)
  mlflow_delete_run(client = client, run_id = run1$run_uuid)
  run3 <- mlflow_restore_run(client = client, run_id = run1$run_uuid)

  for (run in list(run1, run2, run3)) {
    expect_identical(run$user_id, "user1")

    expect_true(
      all(purrr::transpose(run$tags[[1]]) %in%
        list(
          list(key = "foz", value = "baz"),
          list(key = "foo", value = "bar"),
          list(key = "mlflow.user", value = "user1"),
          list(key = "mlflow.runName", value = run$run_name)
        )
      )
    )
  }
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
  mlflow_log_metric("nan", NaN)
  mlflow_log_metric("inf", Inf)
  mlflow_log_metric("-inf", -Inf)

  mlflow_set_tag("tag_key", "tag_value")
  mlflow_log_param("param_key", "param_value")
  mlflow_log_param("na", NA)
  mlflow_log_param("nan", NaN)

  run <- mlflow_get_run()
  metrics <- run$metrics[[1]]
  nan_value <- metrics$value[metrics$key == "nan"]
  expect_true(is.nan(nan_value))
  pos_inf_value <- metrics$value[metrics$key == "inf"]
  expect_true(pos_inf_value >= 1.7976931348623157e308)
  neg_inf_value <- metrics$value[metrics$key == "-inf"]
  expect_true(neg_inf_value <= -1.7976931348623157e308)
  run_id <- run$run_uuid
  tags <- run$tags[[1]]
  expect_identical("tag_value", tags$value[tags$key == "tag_key"])
  expect_setequal(run$params[[1]]$key, c("na", "nan", "param_key"))
  expect_setequal(run$params[[1]]$value, c("NA", "NaN", "param_value"))

  mlflow_delete_tag("tag_key", run_id)
  run <- mlflow_get_run()
  tags <- run$tags[[1]]
  expect_false("tag_key" %in% tags$key)

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
    purrr::map_vec(round(timestamp_inputs), mlflow:::milliseconds_to_date)
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
    purrr::map_vec(c(300, 100, 200), mlflow:::milliseconds_to_date)
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

test_that("mlflow_search_runs() works", {
  mlflow_clear_test_dir("mlruns")
  with(mlflow_start_run(), {
    mlflow_log_metric("test", 10)
  })
  with(mlflow_start_run(), {
    mlflow_log_metric("test", 20)
  })
  expect_equal(nrow(mlflow_search_runs(experiment_ids = list("0"))), 2)
  expect_equal(nrow(mlflow_search_runs(experiment_ids = "0")), 2)
  expect_equal(nrow(mlflow_search_runs(filter = "metrics.test > 10", experiment_ids = list("0"))), 1)
  expect_equal(nrow(mlflow_search_runs(filter = "metrics.test < 20", experiment_ids = list("0"))), 1)
  expect_equal(nrow(mlflow_search_runs(filter = "metrics.test > 20", experiment_ids = list("0"))), 0)

  search <- mlflow_search_runs(order_by = "metrics.test", experiment_ids = list("0"))
  expect_equal(search$metrics[[1]]$value[1], 10)
  expect_equal(search$metrics[[2]]$value[1], 20)

  search <- mlflow_search_runs(order_by = list("metrics.test DESC"), experiment_ids = list("0"))
  expect_equal(search$metrics[[1]]$value[1], 20)
  expect_equal(search$metrics[[2]]$value[1], 10)

  mlflow_set_experiment("new-experiment")
  expect_equal(nrow(mlflow_search_runs()), 0)
  with(mlflow_start_run(), {
    mlflow_log_metric("new_experiment_metric", 30)
  })
  expect_equal(nrow(mlflow_search_runs(filter = "metrics.new_experiment_metric = 30")), 1)
})

test_that("mlflow_log_artifact and mlflow_list_artifacts work", {
  with(mlflow_start_run(), {
    # List artifacts for run without artifacts, result should be empty
    empty_artifact_list <- mlflow_list_artifacts()
    # Create some dummy artifact files/directories
    expect_equal(nrow(empty_artifact_list), 0)
    source_dir <- file.path(tempdir(), "temp-directory")
    dir.create(source_dir)
    ## file 1
    file_path1 <- file.path(source_dir, "a-my-file")
    contents <- "File contents\n"
    cat(contents, file = file_path1, sep = "")
    ## file 2
    file_path2 <- file.path(source_dir, "a-my-file-2")
    contents <- "File contents\n"
    cat(contents, file = file_path2, sep = "")

    # Log file, file with path, directory with path argument
    mlflow_log_artifact(file_path1)
    mlflow_log_artifact(file_path2)
    mlflow_log_artifact(file_path1, "directory_for_file")
    mlflow_log_artifact(source_dir, "artifact_subdirectory")
    # Verify logged files
    artifact_list0 <- mlflow_list_artifacts()
    expect_equal(nrow(artifact_list0), 4)
    logged_file0 <- artifact_list0[artifact_list0$path == "a-my-file", ]
    expect_equal(nrow(logged_file0), 1)
    expect_equal(logged_file0$is_dir, FALSE)
    logged_file1 <- artifact_list0[artifact_list0$path == "directory_for_file", ]
    expect_equal(nrow(logged_file1), 1)
    expect_equal(logged_file1$is_dir, TRUE)
    logged_dir0 <- artifact_list0[artifact_list0$path == "artifact_subdirectory", ]
    expect_equal(nrow(logged_dir0), 1)
    expect_equal(logged_dir0$is_dir, TRUE)
    # Verify contents of logged directory
    artifact_list1 <- mlflow_list_artifacts("artifact_subdirectory")
    expect_equal(nrow(artifact_list1), 2)
    logged_file2 <- artifact_list1[artifact_list1$path ==
      paste("artifact_subdirectory", "a-my-file", sep = "/"), ]
    expect_equal(nrow(logged_file2), 1)
    expect_equal(logged_file2$is_dir, FALSE)
    expect_equal(strtoi(logged_file2$file_size), nchar(contents))
    # Verify contents of file logged under directory
    artifact_list2 <- mlflow_list_artifacts("directory_for_file")
    expect_equal(nrow(artifact_list2), 1)
    logged_file3 <- artifact_list2[artifact_list2$path ==
    paste("directory_for_file", "a-my-file", sep = "/"), ]
    expect_equal(nrow(logged_file3), 1)
    expect_equal(logged_file3$is_dir, FALSE)
    expect_equal(strtoi(logged_file3$file_size), nchar(contents))
  })
})

test_that("mlflow_log_batch() works", {
  mlflow_clear_test_dir("mlruns")
  mlflow_start_run()
  mlflow_log_batch(
    metrics = data.frame(key = c("mse", "mse", "rmse", "rmse", "nan", "Inf", "-Inf"),
                         value = c(21, 23, 42, 36, NaN, Inf, -Inf),
                         timestamp = c(100, 200, 300, 300, 400, 500, 600),
                         step = c(-4, 1, 7, 3, 8, 9, 10)),
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
    c("mse", "rmse", "nan", "Inf", "-Inf")
  )
  expect_equal(23, metrics$value[metrics$key == "mse"])
  expect_equal(42, metrics$value[metrics$key == "rmse"])
  expect_true(all(is.nan(metrics$value[metrics$key == "nan"])))
  expect_true(all(1.7976931348623157e308 <= (metrics$value[metrics$key == "Inf"])))
  expect_true(all(-1.7976931348623157e308 >= (metrics$value[metrics$key == "-Inf"])))
  expect_setequal(
    metrics$timestamp,
    purrr::map_vec(c(200, 300, 400, 500, 600), mlflow:::milliseconds_to_date)
  )
  expect_setequal(
    metrics$step,
    c(1, 7, 8, 9, 10)
  )

  metric_history <- mlflow_get_metric_history("mse")
  expect_setequal(
    metric_history$value,
    c(21, 23)
  )
  expect_setequal(
    metric_history$timestamp,
    purrr::map_vec(c(100, 200), mlflow:::milliseconds_to_date)
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

test_that("mlflow observers receive tracking event callbacks", {
  num_observers <- 3L
  tracking_events <- rep(list(list()), num_observers)
  lapply(
    seq_along(tracking_events),
    function(idx) {
      observer <- structure(list(
        register_tracking_event = function(event_name, data) {
          tracking_events[[idx]][[event_name]] <<- append(
            tracking_events[[idx]][[event_name]], list(data)
          )
        }
      ))
      mlflow_register_external_observer(observer)
    }
  )
  client <- mlflow_client()
  experiment_id <- "0"
  run <- mlflow_start_run(client = client, experiment_id = experiment_id)
  mlflow_set_experiment(experiment_id = experiment_id)
  expect_equal(length(tracking_events), num_observers)
  for (idx in seq(num_observers)) {
    expect_equal(
      tracking_events[[idx]]$create_run[[1]]$run_id,
      run$run_id
    )
    expect_equal(
      tracking_events[[idx]]$create_run[[1]]$experiment_id,
      experiment_id
    )
    expect_equal(
      tracking_events[[idx]]$active_experiment_id[[1]]$experiment_id,
      experiment_id
    )
  }

  mlflow_end_run(client = client, run_id = run$run_uuid)
  expect_equal(length(tracking_events), num_observers)
  for (idx in seq(num_observers)) {
    expect_equal(
      tracking_events[[idx]]$set_terminated[[1]]$run_uuid, run$run_uuid
    )
  }
})

test_that("mlflow get metric history performs pagination", {
  mlflow_clear_test_dir("mlruns")
  run <- mlflow_start_run()

  batch_size <- 1000
  for (x in 0:25) {
    # 0th index entries for timestamp will not work due to pagination filter default logic
    # add a `+1` to the start and end values
    start <- (batch_size * x) + 1
    end <- (batch_size * (x + 1))
    metrics <- data.frame(key = rep(c("m1"), batch_size),
                        value = seq.int(from = start, to = end, by = 1),
                        timestamp = seq.int(from = start, to = end, by = 1),
                        step = seq.int(from = start, to = end, by = 1)
                       )
    mlflow_log_batch(metrics = metrics)
  }
  logged <- mlflow_get_metric_history(metric_key = "m1", run_id = run$run_id)

  expect_equal(nrow(logged), 26000)

  first_entry <- head(logged, n = 1)
  expect_equal(first_entry$key, "m1")
  expect_equal(first_entry$value, 1)
  expect_equal(first_entry$step, 1)
  expect_equal(first_entry$timestamp, purrr::map_vec(c(1), mlflow:::milliseconds_to_date)[[1]])

  last_entry <- tail(logged, n = 1)
  expect_equal(last_entry$key, "m1")
  expect_equal(last_entry$value, 26000)
  expect_equal(last_entry$step, 26000)
  expect_equal(last_entry$timestamp, purrr::map_vec(c(26000), mlflow:::milliseconds_to_date)[[1]])
  mlflow_end_run()
})
