library(mlflow)

# read parameters
column <- mlflow_log_param("column", 1)

# log total rows
mlflow_log_metric("rows", nrow(iris))

# train model
model <- lm(Sepal.Width ~ iris[[column]], iris)

# log models intercept
mlflow_log_metric("intercept", model$coefficients[["(Intercept)"]])
