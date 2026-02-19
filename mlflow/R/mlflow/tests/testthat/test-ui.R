context("UI")

test_that("mlflow can launch the UI", {
  skip_on_travis()

  url <- mlflow_ui()

  response <- httr::GET(url)
  status <- httr::status_code(response)

  if (status != 200L)
  {
    ui_content <- httr::content(response)
    stop("Status ", status, " retrieved with contents: ", ui_content)
  }

  expect_equal(httr::status_code(response), 200)
})
