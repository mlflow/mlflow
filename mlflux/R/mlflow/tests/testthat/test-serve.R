context("Serve")

library("carrier")

wait_for_server_to_start <- function(server_process, port) {
  status_code <- 500
  for (i in 1:5) {
    tryCatch(
      {
        status_code <- httr::status_code(httr::GET(sprintf("http://127.0.0.1:%d/ping/", port)))
      },
      error = function(...) {
        status_code <- 500
      }
    )
    if (status_code != 200) {
      Sys.sleep(5)
    } else {
      break
    }
  }
  if (status_code != 200) {
    write("FAILED to start the server!", stderr())
    error_text <- server_process$read_error()
    stop("Failed to start the server!", error_text)
  }
}

testthat_model_server <- NULL

teardown({
  if (!is.null(testthat_model_server)) {
    testthat_model_server$kill()
  }
})

test_that("mlflow can serve a model function", {
  mlflow_clear_test_dir("model")

  model <- lm(Sepal.Width ~ Sepal.Length + Petal.Width, iris)
  fn <- crate(~ stats::predict(model, .x), model = model)
  mlflow_save_model(fn, path = "model")
  expect_true(dir.exists("model"))
  port <- httpuv::randomPort()
  testthat_model_server <<- processx::process$new(
    "Rscript",
    c(
      "-e",
      sprintf(
        "mlflow::mlflow_rfunc_serve('model', port = %d, browse = FALSE)",
        port
      )
    ),
    supervise = TRUE,
    stdout = "|",
    stderr = "|"
  )
  Sys.sleep(10)
  tryCatch(
    {
      status_code <- httr::status_code(httr::GET(sprintf("http://127.0.0.1:%d/ping/", port)))
    },
    error = function(e) {
      write("FAILED!", stderr())
      error_text <- testthat_model_server$read_error()
      testthat_model_server$kill()
      stop(e$message, ": ", error_text)
    }
  )

  expect_equal(status_code, 200)

  newdata <- iris[1:2, c("Sepal.Length", "Petal.Width")]

  http_prediction <- httr::content(
    httr::POST(
      sprintf("http://127.0.0.1:%d/predict/", port),
      body = jsonlite::toJSON(as.list(newdata))
    )
  )
  if (is.character(http_prediction)) {
    stop(http_prediction)
  }

  expect_equal(
    unlist(http_prediction),
    as.vector(predict(model, newdata)),
    tolerance = 1e-5
  )
})

test_that("mlflow models server api works with R model function", {
  model <- lm(Sepal.Width ~ Sepal.Length + Petal.Width, iris)
  fn <- crate(~ stats::predict(model, .x), model = model)
  mlflow_save_model(fn, path = "model")
  expect_true(dir.exists("model"))
  port <- httpuv::randomPort()
  testthat_model_server <<- mlflow:::mlflow_cli("models", "serve", "-m", "model", "-p", as.character(port),
    background = TRUE
  )
  wait_for_server_to_start(testthat_model_server, port)
  newdata <- iris[1:2, c("Sepal.Length", "Petal.Width")]
  check_prediction <- function(http_prediction) {
    if (is.character(http_prediction)) {
      stop(http_prediction)
    }
    expect_equal(
      unlist(http_prediction),
      as.vector(predict(model, newdata)),
      tolerance = 1e-5
    )
  }
  # json records
  check_prediction(
    httr::content(
      httr::POST(
        sprintf("http://127.0.0.1:%d/invocation/", port),
        httr::content_type("application/json; format=pandas-records"),
        body = jsonlite::toJSON(as.list(newdata))
      )
    )
  )
  newdata_split <- list(
    columns = names(newdata), index = row.names(newdata),
    data = as.matrix(newdata)
  )
  # json split
  for (content_type in c(
    "application/json",
    "application/json; format=pandas-split",
    "application/json-numpy-split"
  )) {
    check_prediction(
      httr::content(
        httr::POST(
          sprintf("http://127.0.0.1:%d/invocation/", port),
          httr::content_type(content_type),
          body = jsonlite::toJSON(newdata_split)
        )
      )
    )
  }
  # csv
  csv_header <- paste(names(newdata), collapse = ", ")
  csv_row_1 <- paste(newdata[1, ], collapse = ", ")
  csv_row_2 <- paste(newdata[2, ], collapse = ", ")
  newdata_csv <- paste(csv_header, csv_row_1, csv_row_2, "", sep = "\n")
  check_prediction(
    httr::content(
      httr::POST(
        sprintf("http://127.0.0.1:%d/invocation/", port),
        httr::content_type("text/csv"),
        body = newdata_csv
      )
    )
  )
})
