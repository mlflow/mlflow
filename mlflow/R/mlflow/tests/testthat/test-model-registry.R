context("Model Registry")

testthat_uc_model_name <- "test_catalog.test_schema.model"

get_mock_client <- function() {
  client <- new_mlflow_client_impl(
    get_host_creds = function() {
      new_mlflow_host_creds(host = "localhost")
    }
  )

  return(client)
}

test_that("mlflow can register a model", {
  with_mocked_bindings(.package = "mlflow",
            mlflow_registry_rest = function(...) {
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
  with_mocked_bindings(
    .package = "mlflow",
    mlflow_registry_rest = function(...) {
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
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
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
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
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
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
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
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
      args <- list(...)
      expect_equivalent(paste(args[1:2], collapse = "/"), "registered-models/delete")
      expect_equal(args$data$name, "test_model")
  }, {
    mock_client <- get_mock_client()

    mlflow_delete_registered_model("test_model", client = mock_client)
  })
})

test_that("mlflow can retrieve a list of registered models without args", {
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
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
      expect_equal(search_result$registered_models, list())
      expect_null(search_result$next_page_token)
  })
})

test_that("mlflow can retrieve a list of registered models with args", {
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
      args <- list(...)
      expect_true(paste(args[1:2], collapse = "/") == "registered-models/search")
      expect_equal(args$verb, "POST")
      expect_equal(args$data$max_results, 5)
      expect_equal(args$data$page_token, "abc")
      expect_equal(args$data$filter, "name LIKE '%foo'")
      expect_equal(
        args$data$order_by, cast_string_list(list("name ASC", "last_updated_timestamp"))
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

test_that("Unity Catalog registered model search uses UC GET shape", {
  mock_client <- get_mock_client()
  mock_client$registry_uri <- list(scheme = "databricks-uc")
  mock_client$registry_client <- mock_client

  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
      args <- list(...)
      expect_equal(paste(args[1:2], collapse = "/"), "registered-models/search")
      expect_equal(args$verb, "GET")
      expect_equal(args$query$max_results, 5)
      expect_equal(args$query$page_token, "abc")
      expect_null(args$data)
      list(
        registered_models = list(list(name = testthat_uc_model_name)),
        next_page_token = "def"
      )
    }, {
      search_result <- mlflow_search_registered_models(
        max_results = 5,
        page_token = "abc",
        client = mock_client
      )
      expect_equal(search_result$registered_models[[1]]$name, testthat_uc_model_name)
      expect_equal(search_result$next_page_token, "def")
    })
})

test_that("Unity Catalog registered model search rejects workspace-only options", {
  mock_client <- get_mock_client()
  mock_client$registry_uri <- list(scheme = "databricks-uc")
  mock_client$registry_client <- mock_client

  expect_error(
    mlflow_search_registered_models(filter = "name LIKE '%foo'", client = mock_client),
    "does not support `filter`",
    fixed = TRUE
  )
  expect_error(
    mlflow_search_registered_models(order_by = list("name ASC"), client = mock_client),
    "does not support `order_by`",
    fixed = TRUE
  )
})

test_that("mlflow can retrieve a list of model versions", {
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
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
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
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
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
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
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
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
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
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
  with_mocked_bindings(.package = "mlflow",
            mlflow_registry_rest = function(...) {
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
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
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

test_that("Unity Catalog model stages fail locally with alias guidance", {
  mock_client <- get_mock_client()
  mock_client$registry_uri <- list(scheme = "databricks-uc")
  mock_client$registry_client <- mock_client

  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) stop("unexpected registry call"), {
      expect_error(
        mlflow_get_latest_versions(name = testthat_uc_model_name, client = mock_client),
        "aliases"
      )
      expect_error(
        mlflow_transition_model_version_stage(
          name = testthat_uc_model_name,
          version = "1",
          stage = "Production",
          client = mock_client
        ),
        "aliases"
      )
  })
})

test_that("mlflow can set model version tag", {
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
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

test_that("mlflow_set_model_version_tag resolves stage with provided client", {
  mock_client <- get_mock_client()
  resolved_client <- NULL
  tagged_version <- NULL

  with_mocked_bindings(.package = "mlflow",
    mlflow_get_latest_versions = function(name, stages = list(), client = NULL) {
      resolved_client <<- client
      list(list(version = "11"))
    },
    mlflow_registry_rest = function(...) {
      args <- list(...)
      tagged_version <<- args$data$version
      list()
    }, {
      mlflow_set_model_version_tag(
        name = "mymodel",
        stage = "Production",
        key = "test_key",
        value = "test_value",
        client = mock_client
      )
    })

  expect_identical(resolved_client, mock_client)
  expect_equal(tagged_version, "11")
})

test_that("registry rest routes to Unity Catalog path prefix when configured", {
  mock_client <- get_mock_client()
  mock_client$registry_uri <- list(scheme = "databricks-uc")
  mock_client$registry_client <- mock_client

  with_mocked_bindings(.package = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_equal(args$path_prefix, "api/2.0/mlflow/unity-catalog")
      list(model_version = list(name = "mymodel", version = "1"))
    }, {
      mlflow_get_model_version("mymodel", version = "1", client = mock_client)
  })
})

test_that("mlflow_register_model delegates to mlflow_create_model_version", {
  calls <- list()
  with_mocked_bindings(.package = "mlflow",
    mlflow_create_model_version = function(name, source, run_id = NULL, tags = NULL, run_link = NULL,
                                           description = NULL, client = NULL) {
      calls[[1]] <<- list(
        name = name,
        source = source,
        run_id = run_id,
        tags = tags,
        description = description,
        client = client
      )
      list(model_version = list(version = "2"))
    }, {
      mock_client <- get_mock_client()
      mlflow_register_model(
        model_uri = "runs:/abc/model",
        name = testthat_uc_model_name,
        run_id = "abc",
        tags = list(owner = "r"),
        description = "desc",
        client = mock_client
      )
    })

  expect_equal(calls[[1]]$name, testthat_uc_model_name)
  expect_equal(calls[[1]]$source, "runs:/abc/model")
  expect_equal(calls[[1]]$run_id, "abc")
  expect_equal(calls[[1]]$tags$owner, "r")
  expect_equal(calls[[1]]$description, "desc")
})

test_that("model alias APIs call registered-models alias endpoint", {
  calls <- list()
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
      args <- list(...)
      calls[[length(calls) + 1]] <<- args
      if (identical(args$verb, "GET")) {
        return(list(model_version = list(version = "7")))
      }
      list()
    }, {
      mock_client <- get_mock_client()
      mlflow_set_registered_model_alias(testthat_uc_model_name, "prod", "7", client = mock_client)
      resolved <- mlflow_get_model_version_by_alias(
        testthat_uc_model_name,
        "prod",
        client = mock_client
      )
      expect_equal((resolved$model_version %||% resolved)$version, "7")
    })

  expect_equal(paste(calls[[1]][1:2], collapse = "/"), "registered-models/alias")
  expect_equal(calls[[1]]$verb, "POST")
  expect_equal(calls[[2]]$verb, "GET")
})

test_that("UC create model version delegates artifact handling to Python MLflow", {
  mock_client <- get_mock_client()
  mock_client$registry_uri <- list(scheme = "databricks-uc")

  python_payload <- NULL
  get_model_version_args <- NULL
  with_mocked_bindings(.package = "mlflow",
    mlflow_python_create_model_version = function(payload, client) {
      python_payload <<- payload
      expect_identical(client, mock_client)
      "8"
    },
    mlflow_get_model_version = function(name, version, client = NULL) {
      get_model_version_args <<- list(name = name, version = version, client = client)
      list(name = name, version = version, status = "READY")
    }, {
      result <- mlflow_create_model_version(
        name = testthat_uc_model_name,
        source = "runs:/rid/model",
        run_id = "rid",
        description = "registered from R",
        tags = list(owner = "mlflow-r"),
        client = mock_client
      )
    })

  expect_equal(result$version, "8")
  expect_equal(python_payload$name, testthat_uc_model_name)
  expect_equal(python_payload$source, "runs:/rid/model")
  expect_equal(python_payload$run_id, "rid")
  expect_equal(python_payload$description, "registered from R")
  expect_equal(python_payload$tags, list(owner = "mlflow-r"))
  expect_equal(get_model_version_args$name, testthat_uc_model_name)
  expect_equal(get_model_version_args$version, "8")
  expect_identical(get_model_version_args$client, mock_client)
})

test_that("Python MLflow create call uses client tracking and registry URIs", {
  mock_client <- get_mock_client()
  mock_client$tracking_uri <- list(raw_uri = "databricks://PROFILE")
  mock_client$registry_uri <- list(raw_uri = "databricks-uc://PROFILE")

  env <- NULL
  args <- NULL
  with_mocked_bindings(.package = "mlflow",
    python_bin = function() "/usr/bin/python",
    mlflow_is_verbose = function() FALSE,
    run = function(command = NULL, args = character(), echo = TRUE, echo_cmd = FALSE,
                   stderr_callback = NULL) {
      env <<- list(
        tracking_uri = Sys.getenv("MLFLOW_TRACKING_URI"),
        registry_uri = Sys.getenv("MLFLOW_REGISTRY_URI")
      )
      args <<- args
      list(stdout = "8\n")
    }, {
      result <- mlflow_python_create_model_version(
        list(name = "model", source = "runs:/rid/model"),
        client = mock_client
      )
    })

  expect_equal(result, "8")
  expect_equal(env$tracking_uri, "databricks://PROFILE")
  expect_equal(env$registry_uri, "databricks-uc://PROFILE")
  expect_equal(args[1], "-c")
  expect_true(grepl("create_model_version", args[2], fixed = TRUE))
})

test_that("UC model version download delegates to Python MLflow artifacts CLI", {
  mock_client <- get_mock_client()

  cli_args <- NULL
  with_mocked_bindings(.package = "mlflow",
    mlflow_cli = function(..., client = NULL, echo = TRUE) {
      cli_args <<- unlist(list(...), use.names = FALSE)
      expect_identical(client, mock_client)
      expect_false(echo)
      list(stdout = "\n/tmp/downloaded-model\n")
    }, {
      result <- mlflow_download_uc_model_version(testthat_uc_model_name, "4", client = mock_client)
    })

  expect_equal(result, "/tmp/downloaded-model")
  expect_equal(cli_args, c(
    "artifacts", "download",
    "--artifact-uri", sprintf("models:/%s/4", testthat_uc_model_name)
  ))
})

test_that("UC model version download errors when Python MLflow returns no path", {
  with_mocked_bindings(.package = "mlflow",
    mlflow_cli = function(..., client = NULL, echo = TRUE) {
      list(stdout = "\n")
    }, {
      expect_error(
        mlflow_download_uc_model_version(testthat_uc_model_name, "4", client = get_mock_client()),
        "did not return a downloaded Unity Catalog model path",
        fixed = TRUE
      )
    })
})
