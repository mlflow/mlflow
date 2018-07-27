model <- lm(Sepal.Width ~ Sepal.Length, iris)

mlflow_save_model(function(data) {
  predict(model, data)
})
