context("Model xgboost")

idx <- withr::with_seed(3809, sample(nrow(mtcars)))
predictors <- c(
  "mpg", "cyl", "disp", "hp", "drat", "wt", "qsec", "vs",
  "gear", "carb"
)
rownames(mtcars) <- NULL
train <- mtcars[idx[1:25], ]
train <- list(data = train[, predictors], label = train$am)
test <- mtcars[idx[26:32], ]
test <- list(data = test[, predictors], label = test$am)

model <- xgboost::xgboost(
  data = as.matrix(train$data), label = train$label, max_depth = 2,
  eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic"
)

testthat_model_dir <- tempfile("model_")

teardown({
  mlflow_clear_test_dir(testthat_model_dir)
})

test_that("mlflow can save model", {
  mlflow_save_model(model, testthat_model_dir)
  expect_true(dir.exists(testthat_model_dir))
  # Test that we can load the model back and score it.
})

test_that("can load model and predict with rfunc backend", {

  loaded_back_model <- mlflow_load_model(testthat_model_dir)
  prediction <- mlflow_predict(loaded_back_model, as.matrix(test$data))
  expect_equal(
    prediction,
    predict(model, xgboost::xgb.DMatrix(as.matrix(test$data)))
  )

})

test_that("can load and predict with python pyfunct and xgboost backend", {
  pyfunc <- reticulate::import("mlflow.pyfunc")
  py_model <- pyfunc$load_model(testthat_model_dir)
  expect_equal(
    as.numeric(py_model$predict(test$data)),
    unname(predict(model, as.matrix(test$data)))
  )

  mlflow.xgboost <- reticulate::import("mlflow.xgboost")
  xgboost_native_model <- mlflow.xgboost$load_model(testthat_model_dir)
  xgboost <- reticulate::import("xgboost")

  expect_equivalent(
    as.numeric(xgboost_native_model$predict(xgboost$DMatrix(test$data))),
    unname(predict(model, as.matrix(test$data)))
  )
})

test_that("Can predict with cli backend", {
  # # Test that we can score this model with pyfunc backend
  temp_in_csv <- tempfile(fileext = ".csv")
  temp_in_json <- tempfile(fileext = ".json")
  temp_in_json_split <- tempfile(fileext = ".json")
  temp_out <- tempfile(fileext = ".json")
  write.csv(test$data, temp_in_csv, row.names = FALSE)
  mlflow_cli(
    "models", "predict", "-m", testthat_model_dir, "-i", temp_in_csv,
    "-o", temp_out, "-t", "csv", "--install-mlflow"
  )
  prediction <- unlist(jsonlite::read_json(temp_out)$predictions)
  expect_true(!is.null(prediction))
  expect_equal(
    prediction,
    predict(model, xgboost::xgb.DMatrix(as.matrix(test$data)))
  )
  # json records
  jsonlite::write_json(list(dataframe_records = test$data), temp_in_json)
  mlflow_cli(
    "models", "predict", "-m", testthat_model_dir, "-i", temp_in_json, "-o", temp_out,
    "-t", "json", "--install-mlflow"
  )
  prediction <- unlist(jsonlite::read_json(temp_out)$predictions)
  expect_true(!is.null(prediction))
  expect_equal(
    prediction,
    unname(predict(model, as.matrix(test$data)))
  )
  # json split
  mtcars_split <- list(
    columns = names(test$data), index = row.names(test$data),
    data = as.matrix(test$data)
  )
  jsonlite::write_json(list(dataframe_split = mtcars_split), temp_in_json_split)
  mlflow_cli(
    "models", "predict", "-m", testthat_model_dir, "-i", temp_in_json_split,
    "-o", temp_out, "-t",
    "json", "--install-mlflow"
  )
  prediction <- unlist(jsonlite::read_json(temp_out)$predictions)
  expect_true(!is.null(prediction))
  expect_equal(
    prediction,
    unname(predict(model, as.matrix(test$data)))
  )
})
