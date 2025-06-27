context("Model")

library("carrier")

testthat_model_name <- basename(tempfile("model_"))

teardown({
  mlflow_clear_test_dir(testthat_model_name)
})

test_that("mlflow model creation time format", {
  mlflow_clear_test_dir(testthat_model_name)
  model <- lm(Sepal.Width ~ Sepal.Length, iris)
  fn <- crate(~ stats::predict(model, .x), model = model)
  model_spec <- mlflow_save_model(fn, testthat_model_name, model_spec = list(
    utc_time_created = mlflow_timestamp()
  ))
  
  expect_true(dir.exists(testthat_model_name))
  expect_match(model_spec$utc_time_created, "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}")
})

test_that("mlflow can save model function", {
  mlflow_clear_test_dir(testthat_model_name)
  model <- lm(Sepal.Width ~ Sepal.Length, iris)
  fn <- crate(~ stats::predict(model, .x), model = model)
  mlflow_save_model(fn, testthat_model_name)
  expect_true(dir.exists(testthat_model_name))
  # Test that we can load the model back and score it.
  loaded_back_model <- mlflow_load_model(testthat_model_name)
  prediction <- mlflow_predict(loaded_back_model, iris)
  expect_equal(
    prediction,
    predict(model, iris)
  )
  # Test that we can score this model with RFunc backend
  temp_in_csv <- tempfile(fileext = ".csv")
  temp_in_json <- tempfile(fileext = ".json")
  temp_in_json_split <- tempfile(fileext = ".json")
  temp_out <- tempfile(fileext = ".json")
  write.csv(iris, temp_in_csv, row.names = FALSE)
  mlflow_cli("models", "predict", "-m", testthat_model_name, "-i", temp_in_csv, "-o", temp_out, "-t", "csv", "--install-mlflow")
  prediction <- unlist(jsonlite::read_json(temp_out))
  expect_true(!is.null(prediction))
  expect_equal(
    prediction,
    unname(predict(model, iris))
  )
  # json records
  jsonlite::write_json(list(dataframe_records = iris), temp_in_json, row.names = FALSE)
  mlflow_cli("models", "predict", "-m", testthat_model_name, "-i", temp_in_json, "-o", temp_out, "-t", "json", "--install-mlflow")
  prediction <- unlist(jsonlite::read_json(temp_out))
  expect_true(!is.null(prediction))
  expect_equal(
    prediction,
    unname(predict(model, iris))
  )
  # json split
  iris_split <- list(
    dataframe_split = list(
      columns = names(iris)[1:4],
      index = row.names(iris),
      data = as.matrix(iris[, 1:4])))
  jsonlite::write_json(iris_split, temp_in_json_split, row.names = FALSE)
  mlflow_cli("models", "predict", "-m", testthat_model_name, "-i", temp_in_json_split, "-o", temp_out, "-t",
             "json", "--install-mlflow")
  prediction <- unlist(jsonlite::read_json(temp_out))
  expect_true(!is.null(prediction))
  expect_equal(
    prediction,
    unname(predict(model, iris))
  )
})

test_that("mlflow can log model and load it back with a uri", {
  with(run <- mlflow_start_run(), {
    model <- structure(
      list(some = "stuff"),
      class = "test"
    )
    predictor <- crate(~ mean(as.matrix(.x)), model = model)
    predicted <- predictor(0:10)
    expect_true(5 == predicted)
    mlflow_log_model(predictor, testthat_model_name)
  })
  runs_uri <- paste("runs:", run$run_uuid, testthat_model_name, sep = "/")
  loaded_model <- mlflow_load_model(runs_uri)
  expect_true(5 == mlflow_predict(loaded_model, 0:10))
  actual_uri <- paste(run$artifact_uri, testthat_model_name, sep = "/")
  loaded_model_2 <- mlflow_load_model(actual_uri)
  expect_true(5 == mlflow_predict(loaded_model_2, 0:10))
  temp_in  <- tempfile(fileext = ".json")
  temp_out  <- tempfile(fileext = ".json")
  jsonlite::write_json(list(dataframe_records=0:10), temp_in)
  mlflow:::mlflow_cli("models", "predict", "-m", runs_uri, "-i", temp_in, "-o", temp_out,
                      "--content-type", "json", "--install-mlflow")
  prediction <- unlist(jsonlite::read_json(temp_out))
  expect_true(5 == prediction)
  mlflow:::mlflow_cli("models", "predict", "-m", actual_uri, "-i", temp_in, "-o", temp_out,
                      "--content-type", "json", "--install-mlflow")
  prediction <- unlist(jsonlite::read_json(temp_out))
  expect_true(5 == prediction)
})

test_that("mlflow log model records correct metadata with the tracking server", {
  with(run <- mlflow_start_run(), {
    print(run$run_uuid[1])
    model <- structure(
      list(some = "stuff"),
      class = "test"
    )
    predictor <- crate(~ mean(as.matrix(.x)), model = model)
    predicted <- predictor(0:10)
    expect_true(5 == predicted)
    mlflow_log_model(predictor, testthat_model_name)
    model_spec_expected <- mlflow_save_model(predictor, "test")
    tags <- mlflow_get_run()$tags[[1]]
    models <- tags$value[which(tags$key == "mlflow.log-model.history")]
    model_spec_actual <- fromJSON(models, simplifyDataFrame = FALSE)[[1]]
    expect_equal(testthat_model_name, model_spec_actual$artifact_path)
    expect_equal(run$run_uuid[1], model_spec_actual$run_id)
    expect_equal(model_spec_expected$flavors, model_spec_actual$flavors)
  })
})

test_that("mlflow can save and load attributes of model flavor correctly", {
  model_name <- basename(tempfile("model_"))
  model <- structure(list(), class = "trivial")
  path <- file.path(tempdir(), model_name)
  mlflow_save_model(model, path = path)
  model <- mlflow_load_model(path)

  expect_equal(attributes(model$flavor)$spec$key1, "value1")
  expect_equal(attributes(model$flavor)$spec$key2, "value2")
})
