context("Params")

test_that("mlflow can read typed command line parameters", {
  mlflow_clear_test_dir("mlruns")

  mlflow_cli(
    "run",
    "examples/",
    "--entry-point",
    "params_example.R",
    "-P", "my_int=10",
    "-P", "my_num=20.0",
    "-P", "my_str=XYZ"
  )

  expect_true(dir.exists("mlruns"))
  expect_true(dir.exists("mlruns/0"))
  expect_true(file.exists("mlruns/0/meta.yaml"))

  run_dir <- file.path("mlruns/0/", dir("mlruns/0/", pattern = "^[a-zA-Z0-9]+$")[[1]])
  params_dir <- dir(file.path(run_dir, "params"))

  expect_true("my_int" %in% params_dir)
  expect_true("my_num" %in% params_dir)
  expect_true("my_str" %in% params_dir)
})

test_that("ml_param() type checking works", {
  expect_identical(mlflow_param("p1", "a", "string"), "a")
  expect_identical(mlflow_param("p2", 42, "integer"), 42L)
  expect_identical(mlflow_param("p3", 42L), 42L)
  expect_identical(mlflow_param("p4", 12), 12)
})
