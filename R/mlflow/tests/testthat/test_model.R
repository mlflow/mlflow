context("Model")

test_that("mlflow can save model function", {
  model <- lm(Sepal.Width ~ Sepal.Length, iris)

  mlflow_save_model(function(data) {
    predict(model, data)
  })
})
