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

test_that("mlflow can get a registered model", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2], collapse = "/") == "registered-models/get")
      expect_equal(args$verb, "GET")
      expect_equal(args$query$name, "test_model")
      return(list(
        registered_model = list(name = "test_model")
      ))
    }, {
      mock_client <- get_mock_client()

      mlflow_get_registered_model("test_model", client = mock_client)
  })
})

test_that("mlflow can rename a registered model", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_equal(paste(args[1:2], collapse = "/"), "registered-models/rename")
      expect_equal(args$verb, "POST")
      expect_equal(args$data$name, "old_model_name")
      expect_equal(args$data$new_name, "new_model_name")

      return(list(
        registered_model = list(name = "new_model_name")
      ))
    }, {
      mock_client <- get_mock_client()
      mlflow_rename_registered_model("old_model_name", "new_model_name",
                                     client = mock_client)
  })
})

test_that("mlflow can update a model", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_equal(paste(args[1:2], collapse = "/"), "registered-models/update")
      expect_equal(args$verb, "PATCH")
      expect_equal(args$data$name, "test_model")
      return(list(
        registered_model = list(
          name = "test_model",
          description = "New Description"
        )
      ))
    }, {
      mock_client <- get_mock_client()
      mlflow_update_registered_model("test_model", "New Description",
                                     client = mock_client)
  })
})

test_that("mlflow can delete a model", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_equivalent(paste(args[1:2], collapse = "/"), "registered-models/delete")
      expect_equal(args$data$name, "test_model")
  }, {
    mock_client <- get_mock_client()

    mlflow_delete_registered_model("test_model", client = mock_client)
  })
})

test_that("mlflow can retrieve a list of registered models without args", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2], collapse = "/") == "registered-models/search")
      expect_equal(args$verb, "POST")

      return(list(
        registered_models = list(),
        next_page_token = NULL
      ))
    }, {
      mock_client <- get_mock_client()
      search_result <- mlflow_search_registered_models(client = mock_client)
      expect_null(search_result$next_page_token)
  })
})

test_that("mlflow can retrieve a list of registered models with args", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2], collapse = "/") == "registered-models/search")
      expect_equal(args$verb, "POST")
      expect_equal(args$data$max_results, 5)
      expect_equal(args$data$page_token, "abc")
      expect_equal(args$data$filter, "name LIKE '%foo'")
      expect_equal(
        args$data$order_by, mlflow:::cast_string_list(list("name ASC", "last_updated_timestamp"))
      )

      return(list(
        registered_models = list(
          list(
            name = "test_model",
            creation_timestamp = 1.6241e+12,
            last_updated_timestamp = 1.6241e+12,
            user_id = "donald.duck"
          )
        ),
        next_page_token = "def"
      ))
    }, {
      mock_client <- get_mock_client()
      search_result <- mlflow_search_registered_models(filter = "name LIKE '%foo'",
                                                       max_results = 5,
                                                       order_by = list(
                                                         "name ASC", "last_updated_timestamp"
                                                       ),
                                                       page_token = "abc",
                                                       client = mock_client)
      expect_equal(search_result$registered_models, list(
        list(
          name = "test_model",
          creation_timestamp = 1.6241e+12,
          last_updated_timestamp = 1.6241e+12,
          user_id = "donald.duck"
        )
      ))
      expect_equal(search_result$next_page_token, "def")
  })
})

test_that("mlflow can retrieve a list of model versions", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2],
                  collapse = "/") == "registered-models/get-latest-versions")
      expect_equal(args$verb, "GET")

      return(list(model_versions = list()))
    }, {
      mock_client <- get_mock_client()
      mlflow_get_latest_versions(name = "mymodel", client = mock_client)
  })
})

test_that("mlflow can retrieve a list of model versions for given stages", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2],
                  collapse = "/") == "registered-models/get-latest-versions")
      expect_equal(args$verb, "GET")
      return(list(model_versions = list()))
    }, {
      mock_client <- get_mock_client()
      mlflow_get_latest_versions(name = "mymodel",
                                 stages=list("Production", "Archived"),
                                 client = mock_client)
  })
})

test_that("mlflow can create a model version", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2],
                  collapse = "/") == "model-versions/create")
      expect_equal(args$verb, "POST")
      return(list(model_version = list(
        name = "mymodel",
        version = 1,
        source = "test_uri"
      )))
    }, {
      mock_client <- get_mock_client()
      mlflow_create_model_version(name = "mymodel",
                                 source="test_uri",
                                 client = mock_client)
  })
})

test_that("mlflow can get a model version", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2],
                  collapse = "/") == "model-versions/get")
      expect_equal(args$verb, "GET")
      return(list(model_version = list(
                name = "mymodel",
                version = 1,
                source = "test_uri"
      )))
    }, {
      mock_client <- get_mock_client()
      mlflow_get_model_version(name = "mymodel",
                               version = 1,
                               client = mock_client)
  })
})

test_that("mlflow can update a model version", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2],
                        collapse = "/") == "model-versions/update")
      expect_equal(args$verb, "PATCH")
      return(list(model_version = list(
                  name = "mymodel",
                  version = 1,
                  description = "New Description"
      )))
    }, {
      mock_client <- get_mock_client()
      mlflow_update_model_version(name = "mymodel",
                                  version = 1,
                                  description = "New Description",
                                  client = mock_client)
  })
})

test_that("mlflow can delete a model version", {
  with_mock(.env = "mlflow",
            mlflow_rest = function(...) {
              args <- list(...)
              expect_true(paste(args[1:2],
                                collapse = "/") == "model-versions/delete")
              expect_equal(args$verb, "DELETE")
              return(list(model_version = list(
                name = "mymodel",
                version = 1,
                source = "test_uri"
              )))
            }, {
              mock_client <- get_mock_client()
              mlflow_delete_model_version(name = "mymodel",
                                       version = 1,
                                       client = mock_client)
            })
})

test_that("mlflow can transition a model", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2],
                  collapse = "/") == "model-versions/transition-stage")
      expect_equal(args$verb, "POST")
      return(list(model_version = list(
                  name = "mymodel",
                  version = 1,
                  source = "test_uri"
      )))
    }, {
      mock_client <- get_mock_client()
      mlflow_transition_model_version_stage(name = "mymodel",
                                            version = 1,
                                            stage = "Production",
                                            client = mock_client)
  })
})

test_that("mlflow can set model version tag", {
  with_mock(.env = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2],
                  collapse = "/") == "model-versions/set-tag")
      expect_equal(args$verb, "POST")
      return(list(model_version = list(
                  name = "mymodel",
                  version = 1,
                  source = "test_uri"
      )))
    }, {
      mock_client <- get_mock_client()
      mlflow_set_model_version_tag(name = "mymodel",
                                   version = 1,
                                   key = "test_key",
                                   value = "test_value",
                                   client = mock_client)
  })
})
