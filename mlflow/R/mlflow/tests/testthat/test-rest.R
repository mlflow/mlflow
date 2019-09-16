context("rest")

library(withr)


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


test_that("429s are retried", {
  responses = list(list(status_code = 429), list(status_code = 429), list(status_code = 200))
  next_id <- 1
  client <- client1 <- mlflow:::mlflow_client("local")
  with_mock(GET = function(...) {
    res <- responses[[next_id]]
    next_id <- next_id
    res
  }, {
    tryCatch({
      mlflow_rest(client, max_rate_limit_interval=0)
      stop("The rest call should have returned 429 and the function should have thrown.")
    })
    x <- mlflow_rest(client, max_rate_limit_interval=2)
    expect_equal(200, x$status_code)
    next_id <- 1
    x <- mlflow_rest(client, max_rate_limit_interval=2)
    expect_equal(200, x$status_code)
    next_id <- 1
    x <- mlflow_rest(client, max_rate_limit_interval=3)
    expect_equal(200, x$status_code)
    next_id <- 1
    tryCatch({
      mlflow_rest(client, max_rate_limit_interval=1)
      stop("The rest call should have returned 429 and the function should have thrown.")
    })
  })
})
