context("Run")

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
})
