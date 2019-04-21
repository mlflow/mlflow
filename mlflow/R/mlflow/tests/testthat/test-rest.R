context("rest")

test_that("user-agent header is set", {
  config <- list()
  config$insecure <- FALSE
  config$username <- NA
  config$password <- NA
  config$token <- NA

  rest_config <- mlflow:::get_rest_config(config)

  expected_user_agent <- paste("mlflow-r-client", packageVersion("mlflow"), sep = "/")
  expect_equal(rest_config$headers$`User-Agent`, expected_user_agent)
  expect_equal(rest_config$config, list())
})

test_that("basic auth is used", {
  config <- list()
  config$insecure <- FALSE
  config$username <- "hello"
  config$password <- "secret"
  config$token <- NA

  rest_config <- mlflow:::get_rest_config(config)

  expect_equal(rest_config$headers$Authorization, "Basic aGVsbG86c2VjcmV0")
})

test_that("token auth is used", {
  config <- list()
  config$insecure <- FALSE
  config$username <- NA
  config$password <- NA
  config$token <- "taken"

  rest_config <- mlflow:::get_rest_config(config)

  expect_equal(rest_config$headers$Authorization, "Bearer taken")
})

test_that("insecure is used", {
  config <- list()
  config$insecure <- TRUE
  config$username <- NA
  config$password <- NA
  config$token <- NA

  rest_config <- mlflow:::get_rest_config(config)

  expect_equal(rest_config$config, httr::config(ssl_verifypeer = 0, ssl_verifyhost = 0))
})
