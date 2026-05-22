context("Model")

library("carrier")

testthat_model_name <- basename(tempfile("model_"))
testthat_uc_model_name <- "test_catalog.test_schema.model"

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
  mlflow_cli("models", "predict", "-m", testthat_model_name, "-i", temp_in_csv, "-o", temp_out, "-t", "csv", "--env-manager", "uv", "--install-mlflow")
  prediction <- unlist(jsonlite::read_json(temp_out))
  expect_true(!is.null(prediction))
  expect_equal(
    prediction,
    unname(predict(model, iris))
  )
  # json records
  jsonlite::write_json(list(dataframe_records = iris), temp_in_json, row.names = FALSE)
  mlflow_cli("models", "predict", "-m", testthat_model_name, "-i", temp_in_json, "-o", temp_out, "-t", "json", "--env-manager", "uv", "--install-mlflow")
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
             "json", "--env-manager", "uv", "--install-mlflow")
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
  mlflow_cli("models", "predict", "-m", runs_uri, "-i", temp_in, "-o", temp_out,
                      "--content-type", "json", "--env-manager", "uv", "--install-mlflow")
  prediction <- unlist(jsonlite::read_json(temp_out))
  expect_true(5 == prediction)
  mlflow_cli("models", "predict", "-m", actual_uri, "-i", temp_in, "-o", temp_out,
                      "--content-type", "json", "--env-manager", "uv", "--install-mlflow")
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

test_that("mlflow_log_model supports signature parameter", {
  lm_model <- lm(Sepal.Width ~ Sepal.Length, iris)
  model <- crate(~ stats::predict(lm_model, .x), lm_model = lm_model)
  signature <- list(
    inputs = list(x = "double"),
    outputs = list(y = "double")
  )
  logged_path <- NULL
  logged_model_spec <- NULL

  with_mocked_bindings(.package = "mlflow",
    mlflow_log_artifact = function(path, artifact_path = NULL, run_id = NULL, client = NULL) {
      logged_path <<- path
      invisible(path)
    },
    mlflow_get_active_run_id_or_start_run = function() "run-123",
    mlflow_record_logged_model = function(model_spec, run_id = NULL, client = NULL) {
      logged_model_spec <<- model_spec
      invisible(NULL)
    }, {
      mlflow_log_model(model, "signed_model", signature = signature)
      spec <- yaml::read_yaml(file.path(logged_path, "MLmodel"))
      inputs <- jsonlite::fromJSON(spec$signature$inputs, simplifyDataFrame = FALSE)
      outputs <- jsonlite::fromJSON(spec$signature$outputs, simplifyDataFrame = FALSE)
      expect_equal(inputs[[1]]$name, "x")
      expect_equal(inputs[[1]]$type, "double")
      expect_true(inputs[[1]]$required)
      expect_equal(outputs[[1]]$name, "y")
      expect_equal(outputs[[1]]$type, "double")
      expect_true(outputs[[1]]$required)
    })
  expect_type(logged_model_spec$signature$inputs, "character")
  expect_type(logged_model_spec$signature$outputs, "character")
})

test_that("signature normalization validates nested schema shapes", {
  signature <- list(
    inputs = list(
      payload = list(
        type = "object",
        properties = list(
          score = "double",
          tags = list(type = "array", items = "string")
        )
      )
    ),
    outputs = list(
      prediction = list(type = "map", values = "double", required = FALSE)
    )
  )

  normalized <- mlflow_normalize_signature(signature)
  inputs <- jsonlite::fromJSON(normalized$inputs, simplifyDataFrame = FALSE)
  outputs <- jsonlite::fromJSON(normalized$outputs, simplifyDataFrame = FALSE)

  expect_equal(inputs[[1]]$name, "payload")
  expect_equal(inputs[[1]]$type, "object")
  expect_equal(inputs[[1]]$properties$score$type, "double")
  expect_true(inputs[[1]]$properties$score$required)
  expect_equal(inputs[[1]]$properties$tags$type, "array")
  expect_equal(inputs[[1]]$properties$tags$items$type, "string")

  expect_equal(outputs[[1]]$name, "prediction")
  expect_equal(outputs[[1]]$type, "map")
  expect_equal(outputs[[1]]$values$type, "double")
  expect_false(outputs[[1]]$required)
})

test_that("signature normalization rejects invalid nested schema shapes", {
  expect_error(
    mlflow_normalize_signature(list(inputs = list(x = list(type = "unknown")))),
    "Unsupported signature type `unknown`",
    fixed = TRUE
  )
  expect_error(
    mlflow_normalize_signature(list(inputs = list(x = list(type = "array")))),
    "must include `items`",
    fixed = TRUE
  )
  expect_error(
    mlflow_normalize_signature(list(inputs = list(x = list(type = "object", properties = list())))),
    "must include named `properties`",
    fixed = TRUE
  )
  expect_error(
    mlflow_normalize_signature(list(inputs = list(x = list(type = "double", required = "yes")))),
    "invalid `required` value",
    fixed = TRUE
  )
})

test_that("mlflow_log_model rejects data.frame signature fields", {
  lm_model <- lm(Sepal.Width ~ Sepal.Length, iris)
  model <- crate(~ stats::predict(lm_model, .x), lm_model = lm_model)

  expect_error(
    mlflow_log_model(
      model,
      "signed_model",
      signature = data.frame(inputs = "x", outputs = "y")
    ),
    "`signature` must be a named list, not a data.frame",
    fixed = TRUE
  )

  expect_error(
    mlflow_log_model(
      model,
      "signed_model",
      signature = list(
        inputs = data.frame(name = "x", type = "double"),
        outputs = list(y = "double")
      )
    ),
    "must be a named list, not a data.frame"
  )

  expect_error(
    mlflow_log_model(model, "signed_model", signature = list()),
    "`signature` must include `inputs` or `outputs`",
    fixed = TRUE
  )
})

test_that("models URI parser supports aliases, versions, and stages", {
  expect_true(mlflow_is_plain_models_uri(
    paste0("models:/", testthat_uc_model_name, "/12")
  ))
  expect_false(mlflow_is_plain_models_uri(
    paste0("models://profile/", testthat_uc_model_name, "/12")
  ))

  alias_uri <- mlflow_parse_models_uri(paste0("models:/", testthat_uc_model_name, "@champion"))
  expect_equal(alias_uri$name, testthat_uc_model_name)
  expect_null(alias_uri$version)
  expect_null(alias_uri$stage)
  expect_equal(alias_uri$alias, "champion")

  version_uri <- mlflow_parse_models_uri(paste0("models:/", testthat_uc_model_name, "/12"))
  expect_equal(version_uri$name, testthat_uc_model_name)
  expect_equal(version_uri$version, "12")
  expect_null(version_uri$stage)
  expect_null(version_uri$alias)

  stage_uri <- mlflow_parse_models_uri("models:/workspace_model/Staging")
  expect_equal(stage_uri$name, "workspace_model")
  expect_null(stage_uri$version)
  expect_equal(stage_uri$stage, "Staging")
  expect_null(stage_uri$alias)

  expect_error(
    mlflow_parse_models_uri("models:/workspace_model@"),
    "Model alias URIs"
  )
})

test_that("models URI alias resolves via registry helper", {
  model_name <- basename(tempfile("model_"))
  lm_model <- lm(Sepal.Width ~ Sepal.Length, iris)
  model <- crate(~ stats::predict(lm_model, .x), lm_model = lm_model)
  path <- file.path(tempdir(), model_name)
  mlflow_save_model(model, path = path)

  with_mocked_bindings(.package = "mlflow",
    mlflow_get_model_version_by_alias = function(name, alias, client = NULL) {
      list(model_version = list(version = "9"))
    },
    mlflow_get_model_version_download_uri = function(name, version, client = NULL) {
      path
    }, {
      loaded <- mlflow_load_model(paste0("models:/", testthat_uc_model_name, "@prod"))
      pred <- mlflow_predict(loaded, iris[1:3, ])
      expect_equal(as.numeric(pred), as.numeric(stats::predict(lm_model, iris[1:3, ])))
    })
})

test_that("Unity Catalog models URI alias downloads via UC helper", {
  model_name <- basename(tempfile("model_"))
  lm_model <- lm(Sepal.Width ~ Sepal.Length, iris)
  model <- crate(~ stats::predict(lm_model, .x), lm_model = lm_model)
  path <- file.path(tempdir(), model_name)
  mlflow_save_model(model, path = path)

  mock_client <- new_mlflow_client_impl(
    get_host_creds = function() {
      new_mlflow_host_creds(host = "https://adb.example.com", token = "x")
    },
    get_cli_env = list
  )
  mock_client$registry_uri <- list(scheme = "databricks-uc")

  downloaded <- NULL
  with_mocked_bindings(.package = "mlflow",
    mlflow_get_model_version_by_alias = function(name, alias, client = NULL) {
      list(model_version = list(version = "9"))
    },
    mlflow_download_uc_model_version = function(name, version, client = NULL) {
      downloaded <<- list(name = name, version = version, client = client)
      path
    }, {
      loaded <- mlflow_load_model(
        paste0("models:/", testthat_uc_model_name, "@prod"),
        client = mock_client
      )
      pred <- mlflow_predict(loaded, iris[1:3, ])
      expect_equal(as.numeric(pred), as.numeric(stats::predict(lm_model, iris[1:3, ])))
    })

  expect_equal(downloaded$name, testthat_uc_model_name)
  expect_equal(downloaded$version, "9")
  expect_identical(downloaded$client, mock_client)
})

test_that("Unity Catalog models URI stages fail with alias guidance", {
  mock_client <- new_mlflow_client_impl(
    get_host_creds = function() {
      new_mlflow_host_creds(host = "https://adb.example.com", token = "x")
    },
    get_cli_env = list
  )
  mock_client$registry_uri <- list(scheme = "databricks-uc")

  expect_error(
    mlflow_download_model_uri(
      paste0("models:/", testthat_uc_model_name, "/Staging"),
      client = mock_client
    ),
    "aliases"
  )
})
