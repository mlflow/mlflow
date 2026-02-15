context("Params")

test_that("mlflow can read typed command line parameters", {
  mlflow_clear_test_dir()

  mlflow_cli(
    "run",
    "examples/",
    "--env-manager",
    "uv",
    "--entry-point",
    "params_example.R",
    "-P", "my_int=10",
    "-P", "my_num=20.0",
    "-P", "my_str=XYZ"
  )

  # With SQLite backend, we no longer check for mlruns directory structure
  # Instead, verify the run was logged by searching for runs
  client <- mlflow_client()
  runs <- mlflow_search_runs(experiment_ids = list("0"), client = client)
  expect_true(nrow(runs) > 0)
  
  # Verify params were logged
  if (nrow(runs) > 0) {
    run <- runs[1, ]
    params <- run$params[[1]]
    expect_true("my_int" %in% params$key)
    expect_true("my_num" %in% params$key)
    expect_true("my_str" %in% params$key)
  }
})

test_that("ml_param() type checking works", {
  expect_identical(mlflow_param("p1", "a", "string"), "a")
  expect_identical(mlflow_param("p2", 42, "integer"), 42L)
  expect_identical(mlflow_param("p3", 42L), 42L)
  expect_identical(mlflow_param("p4", 12), 12)
})
