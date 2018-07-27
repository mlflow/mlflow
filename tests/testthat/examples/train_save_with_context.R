model <- lm(Sepal.Width ~ Sepal.Length, iris)

one <- 1
mlflow_save_model(function(data) {
  predict(model, data) + one
})
