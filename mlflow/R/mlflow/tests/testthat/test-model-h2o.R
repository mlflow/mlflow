context("Model h2o")

setup({
  h2o::h2o.init(port = httpuv::randomPort())
})

idx <- withr::with_seed(3809, sample(nrow(iris)))
prediction <- "Species"
predictors <- setdiff(colnames(iris), prediction)
train <- iris[idx[1:100], ]
test <- iris[idx[101:nrow(iris)], ]

# Installing most recent h2o package, see https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html#install-in-r
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}
# Pin h2o to prevent version-mismatch between python and R
install.packages("https://cran.r-project.org/src/contrib/Archive/h2o/h2o_3.30.1.3.tar.gz", repos=NULL, type="source")

h2o::h2o.init()

model <- h2o::h2o.randomForest(
  x = predictors, y = prediction, training_frame = h2o::as.h2o(train)
)
testthat_model_dir <- tempfile("model_")

teardown({
  h2o::h2o.shutdown(prompt = FALSE)
  mlflow_clear_test_dir(testthat_model_dir)
})

test_that("mlflow can save model", {
  mlflow_save_model(model, testthat_model_dir)
  expect_true(dir.exists(testthat_model_dir))
})

test_that("can load model and predict with rfunc backend", {
  saved_model <- mlflow_load_model(testthat_model_dir)
  prediction <- mlflow_predict(saved_model, test)
  expect_equal(
    prediction,
    as.data.frame(h2o::h2o.predict(model, h2o::as.h2o(test)))
  )
})

test_that("can print model correctly after it is loaded", {
  saved_model <- mlflow_load_model(testthat_model_dir)
  expect_equal(capture_output(print(model)), capture_output(print(saved_model)))
})

test_that("can load and predict with python pyfunct and h2o backend", {
  pyfunc <- import("mlflow.pyfunc")
  py_model <- pyfunc$load_model(testthat_model_dir)

  expected <- as.data.frame(h2o::h2o.predict(model, h2o::as.h2o(test)))
  expected$predict <- as.character(expected$predict)

  expect_equivalent(
    as.data.frame(py_model$predict(test)), expected
  )

  mlflow.h2o <- import("mlflow.h2o")
  h2o_native_model <- mlflow.h2o$load_model(testthat_model_dir)
  h2o <- import("h2o")

  expect_equivalent(
    as.data.frame(
      h2o_native_model$predict(h2o$H2OFrame(test))$as_data_frame()
    ),
    expected
  )
})

test_that("Can predict with cli backend", {
  expected <- as.data.frame(h2o::h2o.predict(model, h2o::as.h2o(test)))

  # # Test that we can score this model with pyfunc backend
  temp_in_csv <- tempfile(fileext = ".csv")
  temp_in_json <- tempfile(fileext = ".json")
  temp_in_json_split <- tempfile(fileext = ".json")
  temp_out <- tempfile(fileext = ".json")

  check_output <- function() {
    actual <- do.call(
      rbind,
      lapply(jsonlite::read_json(temp_out), as.data.frame)
    )

    expect_true(!is.null(actual))
    actual$predict <- as.factor(actual$predict)
    expect_equal(actual, expected)
  }

  write.csv(test[, predictors], temp_in_csv, row.names = FALSE)
  mlflow_cli(
    "models", "predict", "-m", testthat_model_dir, "-i", temp_in_csv,
    "-o", temp_out, "-t", "csv"
  )
  check_output()

  # json records
  jsonlite::write_json(test[, predictors], temp_in_json)
  mlflow_cli(
    "models", "predict", "-m", testthat_model_dir, "-i", temp_in_json, "-o", temp_out,
    "-t", "json",
    "--json-format", "records"
  )
  check_output()

  # json split
  mtcars_split <- list(
    columns = colnames(test), index = row.names(test), data = as.matrix(test)
  )
  jsonlite::write_json(mtcars_split, temp_in_json_split)
  mlflow_cli(
    "models", "predict", "-m", testthat_model_dir, "-i", temp_in_json_split,
    "-o", temp_out, "-t",
    "json", "--json-format", "split"
  )
  check_output()
})
