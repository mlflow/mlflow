context("Model mleap")

library(mleap)

sc <- sparklyr::spark_connect(master = "local", version = "2.4.5")
testthat_model_dir <- basename(tempfile("model_"))

teardown({
  sparklyr::spark_disconnect(sc)
  mlflow_clear_test_dir(testthat_model_dir)
})

mtcars_sdf <- sparklyr::copy_to(sc, mtcars, overwrite = TRUE)

pipeline <- sparklyr::ml_pipeline(sc) %>%
  sparklyr::ft_binarizer("hp", "high_hp", threshold = 100) %>%
  sparklyr::ft_vector_assembler(c("high_hp", "wt", "qsec"), "features") %>%
  sparklyr::ml_gbt_regressor(label_col = "mpg")

model <- sparklyr::ml_fit(pipeline, mtcars_sdf)

test_that("mlflow can save model", {
  mlflow_save_model(model, testthat_model_dir, sample_input = mtcars_sdf)
  expect_true(dir.exists(testthat_model_dir))
})

test_that("can load model and predict with `mlflow_predict()`", {
  mleap_transformer <- mlflow_load_model(testthat_model_dir)
  input <- mtcars[c("qsec", "hp", "wt")]
  predictions <- mlflow_predict(mleap_transformer, input)

  expect_equal(nrow(predictions), nrow(mtcars))
  expect_true("high_hp" %in% colnames(predictions))
  expect_true("prediction" %in% colnames(predictions))
  # at the moment a GBT model can fit `mtcars` perfectly because it simply
  # "memorized" the dataset
  expect_equal(ifelse(mtcars$hp >= 100, 1, 0), predictions$high_hp)
})

test_that("can load model created by MLflow Java client and predict with `mlflow_predict()`", {
  model_dir <- file.path(
    "..", "..", "..", "..", "java", "scoring", "src", "test", "resources", "org", "mlflow", "mleap_model"
  )
  model <- mlflow_load_model(model_dir)

  input <- jsonlite::fromJSON(file.path(model_dir, "sample_input.json"))
  data <- as.data.frame(input$data)
  colnames(data) <- input$columns
  predictions <- mlflow_predict(model, data)

  expect_equal(
    colnames(predictions),
    c("text", "topic", "label", "words", "features", "rawPrediction", "probability", "prediction")
  )
})
