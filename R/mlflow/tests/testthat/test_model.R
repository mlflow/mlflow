context("Model")

test_that("mlflow can save model function", {
  mlflow_clear_test_dir("model")

  model <- lm(Sepal.Width ~ Sepal.Length, iris)

  mlflow_save_model(function(data, model) {
    predict(model, data)
  })

  expect_true(dir.exists("model"))

  prediction <- mlflow_rfunc_predict("model", iris)
  expect_true(!is.null(prediction))

  expect_equal(
    prediction,
    predict(model, iris)
  )
})
