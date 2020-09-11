context("Model h2o")

idx <- withr::with_seed(3809, sample(nrow(iris)))
predictor <- "Species"

train <- iris[idx[1:100], ]
test <- iris[idx[101:nrow(iris)], ]

h2o::h2o.init()

model <- h2o::h2o.randomForest(
  x = setdiff(colnames(iris), predictor), y = predictor, training_frame = h2o::as.h2o(train)
)

test_that("mlflow can save model", {
  mlflow_clear_test_dir("model")

  mlflow_save_model(model, "model")
  expect_true(dir.exists("model"))
})

test_that("can load model and predict with rfunc backend", {
  saved_model <- mlflow_load_model("model")
  prediction <- mlflow_predict(saved_model, test)
  expect_equal(
    prediction,
    as.data.frame(h2o::h2o.predict(model, h2o::as.h2o(test)))
  )
})

test_that("can load and predict with python pyfunct and h2o backend", {
  pyfunc <- import("mlflow.pyfunc")
  py_model <- pyfunc$load_model("model")

  expected <- as.data.frame(h2o::h2o.predict(model, h2o::as.h2o(test)))
  expected$predict <- as.character(expected$predict)

  expect_equivalent(
    as.data.frame(py_model$predict(test)), expected
  )

  mlflow.h2o <- import("mlflow.h2o")
  h2o_native_model <- mlflow.h2o$load_model("model")
  h2o <- import("h2o")

  expect_equivalent(
    as.data.frame(
      h2o_native_model$predict(h2o$H2OFrame(test))$as_data_frame()
    ),
    expected
  )
})
