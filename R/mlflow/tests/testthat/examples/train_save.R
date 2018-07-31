library(mlflow)

# train model
model <- lm(Sepal.Width ~ Sepal.Length, iris)

# save model
mlflow_save_model(function(data) {
  predict(model, data)
})
