context("Model")

test_that("mlflow can save model function", {
  model <- lm(Sepal.Width ~ Sepal.Length, iris)

  mlflow_save_model(function(data, model) {
    predict(model, data)
  })

  model <- mlflow_load_model("model")
  mlflow_predict_model(model, iris)
})
