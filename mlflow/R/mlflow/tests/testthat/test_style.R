context("lints")

test_that("mlflow package conforms to lintr::lint_package() style", {
  if (nchar(Sys.getenv("COVR_RUNNING")) > 0) {
    skip("No linting during code coverage")
  }

  lintr::expect_lint_free()
})
