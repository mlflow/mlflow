context("Serve")

test_that("mlflow can serve a model function", {
  mlflow_clear_test_dir("model")

  model <- lm(Sepal.Width ~ Sepal.Length, iris)

  fn <- crate(~ stats::predict(model, .x), model)
  mlflow_save_model(fn)

  expect_true(dir.exists("model"))

  model_server <- processx::process$new(
    "Rscript",
    c(
      "-e",
      "mlflow::mlflow_rfunc_serve('model', browse = FALSE)"
    ),
    supervise = TRUE,
    stdout = "|",
    stderr = "|"
  )
  on.exit(model_server$kill())

  Sys.sleep(5)

  status_code <- httr::status_code(httr::GET("http://127.0.0.1:8090"))
  expect_equal(status_code, 200)

  http_prediction <- httr::content(
    httr::POST(
      "http://127.0.0.1:8090/predict/",
      body = jsonlite::toJSON(iris[1,])
    )
  )
  if (is.character(http_prediction)) {
    stop(http_prediction)
  }

  expect_equal(
    unlist(http_prediction),
    as.vector(predict(model, iris[1,])),
    tolerance = 1e-5
  )
})
