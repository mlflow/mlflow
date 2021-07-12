context("Model Registry")

get_mock_client <- function() {
  client <- new_mlflow_client_impl(
    get_host_creds = function() {
      new_mlflow_host_creds(host = "localhost")
    }
  )

  return(client)
}

test_that("mlflow can register a model", {
  with_mock(.env = "mlflow",
            mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2], collapse = "/") == "registered-models/create")

      data <- args$data
      expect_equal(data$name, "test_model")

      return(list(
        registered_model = list(
          name = "test_model",
          creation_timestamp = 1.6241e+12,
          last_updated_timestamp = 1.6241e+12,
          user_id = "donald.duck"
        )
      ))
    }, {
      mock_client <- get_mock_client()
      registered_model <- mlflow_create_registered_model("test_model", client = mock_client)

      expect_true("name" %in% names(registered_model))
      expect_true("creation_timestamp" %in% names(registered_model))
      expect_true("last_updated_timestamp" %in% names(registered_model))
      expect_true("user_id" %in% names(registered_model))
    })
})

test_that("mlflow can register a model with tags and description", {
  with_mock(
    .env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2], collapse = "/") == "registered-models/create")

      data <- args$data
      expect_equal(data$name, "test_model")
      expect_equal(data$description, "Some test model")

      return(list(
        registered_model = list(
          name = "test_model",
          creation_timestamp = 1.6241e+12,
          last_updated_timestamp = 1.6241e+12,
          user_id = "donald.duck",
          tags = list(list(
            key = "creator", value = "Donald Duck"
          )),
          description = "Some test model"
        )
      ))
    }, {
      mock_client <- get_mock_client()

      registered_model <- mlflow_create_registered_model(
          "test_model",
          tags = list(list(key = "creator", value = "Donald Duck")),
          description = "Some test model",
          client = mock_client
        )
      expect_equal(length(registered_model$tags), 1)
    }
  )
})

test_that("mlflow can delete a model", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2], collapse = "/") == "registered-models/delete")
      expect_equal(args$data$name, "test_model")
  }, {
    mock_client <- get_mock_client()

    mlflow_delete_registered_model("test_model", client = mock_client)
  })
})
