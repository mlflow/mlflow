# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

library(mlflow)
library(glmnet)
library(carrier)

set.seed(40)

# Read the wine-quality csv file
data <- read.csv("wine-quality.csv")

# Split the data into training and test sets. (0.75, 0.25) split.
sampled <- sample(1:nrow(data), 0.75 * nrow(data))
train <- data[sampled, ]
test <- data[-sampled, ]

# The predicted column is "quality" which is a scalar from [3, 9]
train_x <- as.matrix(train[, !(names(train) == "quality")])
test_x <- as.matrix(test[, !(names(train) == "quality")])
train_y <- train[, "quality"]
test_y <- test[, "quality"]

alpha <- mlflow_param("alpha", 0.5, "numeric")
lambda <- mlflow_param("lambda", 0.5, "numeric")

with(mlflow_start_run(), {
    model <- glmnet(train_x, train_y, alpha = alpha, lambda = lambda, family= "gaussian", standardize = FALSE)
    predictor <- crate(~ glmnet::predict.glmnet(!!model, as.matrix(.x)), !!model)
    predicted <- predictor(test_x)

    rmse <- sqrt(mean((predicted - test_y) ^ 2))
    mae <- mean(abs(predicted - test_y))
    r2 <- as.numeric(cor(predicted, test_y) ^ 2)

    message("Elasticnet model (alpha=", alpha, ", lambda=", lambda, "):")
    message("  RMSE: ", rmse)
    message("  MAE: ", mae)
    message("  R2: ", r2)

    mlflow_log_param("alpha", alpha)
    mlflow_log_param("lambda", lambda)
    mlflow_log_metric("rmse", rmse)
    mlflow_log_metric("r2", r2)
    mlflow_log_metric("mae", mae)

    mlflow_log_model(predictor, "model")
})
