context("Model")

test_that("mlflow can save model function", {
  mlflow_clear_test_dir("model")
  model <- lm(Sepal.Width ~ Sepal.Length, iris)
  fn <- crate(~ stats::predict(model, .x), model)
  mlflow_save_model(fn, "model")
  expect_true(dir.exists("model"))
  temp_in <- tempfile(fileext = ".csv")
  temp_out <- tempfile(fileext = ".csv")
  write.csv(iris, temp_in, row.names = FALSE)
  mlflow_rfunc_predict("model", input_path = temp_in, output_path = temp_out)
  prediction <- read.csv(temp_out)[[1]]

  expect_true(!is.null(prediction))

  expect_equal(
    prediction,
    unname(predict(model, iris))
  )
})

test_that("mlflow can write model with dependencies", {
  mlflow_clear_test_dir("model")
  model <- lm(Sepal.Width ~ Sepal.Length, iris)
  fn <- crate(~ stats::predict(model, .x), model)
  mlflow_save_model(fn, "model", conda_env = "conda.yaml")
  mlmodel <- yaml::read_yaml("model/MLmodel")
  expect_equal(
    mlmodel$flavors$crate$conda_env,
    "conda.yaml"
  )
})

test_that("mlflow can log model and load it back with a uri", {
  with(run <- mlflow_start_run(), {
    model <- structure(
      list(some = "stuff"),
      class = "test"
    )
    predictor <- crate(~ mean(as.matrix(.x)), model)
    predicted <- predictor(0:10)
    expect_true(5 == predicted)
    mlflow_log_model(predictor, "model")
  })
  runs_uri <- paste("runs:", run$run_uuid, "model", sep = "/")
  loaded_model <- mlflow_load_model(runs_uri)
  expect_true(5 == mlflow_predict_flavor(loaded_model, 0:10))
  actual_uri <- paste(run$artifact_uri, "model", sep = "/")
  loaded_model_2 <- mlflow_load_model(actual_uri)
  expect_true(5 == mlflow_predict_flavor(loaded_model_2, 0:10))
  expect_true(5 == mlflow_rfunc_predict(runs_uri, data = 0:10))
  expect_true(5 == mlflow_rfunc_predict(actual_uri, data = 0:10))
})
