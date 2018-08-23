context("lints")

test_that("mlflow package conforms to lintr::lint_package() style", {
  lintr::expect_lint_free()
})
