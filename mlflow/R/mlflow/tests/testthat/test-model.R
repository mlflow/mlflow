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
