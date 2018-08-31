context("Model")

test_that("mlflow can save model function", {
  mlflow_clear_test_dir("model")

  model <- lm(Sepal.Width ~ Sepal.Length, iris)

  fn <- crate(~ stats::predict(model, .x), model)

  mlflow_log_model(fn, "model")

  expect_true(dir.exists("model"))

  prediction <- mlflow_rfunc_predict("model", data = iris)
  expect_true(!is.null(prediction))

  expect_equal(
    prediction,
    predict(model, iris)
  )
})
