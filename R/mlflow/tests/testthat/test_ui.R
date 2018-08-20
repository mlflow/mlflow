context("UI")

test_that("mlflow can launch the UI", {
  url <- mlflow_ui()

  response <- httr::GET(url)
  expect_equal(httr::status_code(response), 200)
})
