context("Model")

library("keras")

test_that("mlflow can save keras model ", {
  mlflow_clear_test_dir("model")
  model <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = "relu", input_shape = dim(iris)[2] - 1) %>%
  layer_dense(units = 3, activation = "softmax")
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )
  train_x <- as.matrix(iris[, 1:4])
  train_y <- to_categorical(as.numeric(iris[, 5]) - 1, 3)
  model %>% fit(train_x, train_y, epochs = 1)
  model %>% mlflow_save_model("model")
  expect_true(dir.exists("model"))

  model_reloaded <- mlflow_load_model("model")

  expect_equal(
    model %>% predict(train_x),
    model_reloaded %>% predict(train_x),
    mlflow_predict_flavor(model_reloaded, iris[, 1:4])
  )
})
