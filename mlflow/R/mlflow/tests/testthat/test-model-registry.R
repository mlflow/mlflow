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
        registered_models = list(list(name = "zacdav.default.m")),
        next_page_token = "def"
      )
    }, {
      search_result <- mlflow_search_registered_models(
        max_results = 5,
        page_token = "abc",
        client = mock_client
      )
      expect_equal(search_result$registered_models[[1]]$name, "zacdav.default.m")
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
        mlflow_get_latest_versions(name = "zacdav.default.m", client = mock_client),
        "aliases"
      )
      expect_error(
        mlflow_transition_model_version_stage(
          name = "zacdav.default.m",
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
        name = "zacdav.default.m",
        run_id = "abc",
        tags = list(owner = "r"),
        description = "desc",
        client = mock_client
      )
    })

  expect_equal(calls[[1]]$name, "zacdav.default.m")
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
      mlflow_set_registered_model_alias("zacdav.default.m", "prod", "7", client = mock_client)
      resolved <- mlflow_get_model_version_by_alias("zacdav.default.m", "prod", client = mock_client)
      expect_equal((resolved$model_version %||% resolved)$version, "7")
    })

  expect_equal(paste(calls[[1]][1:2], collapse = "/"), "registered-models/alias")
  expect_equal(calls[[1]]$verb, "POST")
  expect_equal(calls[[2]]$verb, "GET")
})

test_that("UC model source materialization is explicit", {
  model_dir <- tempfile("uc-local-model-")
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(model_dir, recursive = TRUE), add = TRUE)

  expect_equal(
    mlflow_materialize_local_model(model_dir, client = get_mock_client()),
    normalizePath(model_dir, winslash = "/", mustWork = TRUE)
  )
})

test_that("UC source materialization does not hide generic download errors", {
  with_mocked_bindings(.package = "mlflow",
    mlflow_download_artifacts_from_uri = function(artifact_uri, client = mlflow_client()) {
      stop("download failed", call. = FALSE)
    }, {
      expect_error(
        mlflow_materialize_local_model("models:/missing/1", client = get_mock_client()),
        "download failed"
      )
    })
})

test_that("UC create model version calls create credentials upload finalize", {
  model_dir <- file.path(tempdir(), paste0("uc-model-", as.integer(Sys.time())))
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(model_dir, recursive = TRUE), add = TRUE)

  yaml::write_yaml(
    list(
      signature = list(
        inputs = list(list(name = "x", type = "double")),
        outputs = list(list(name = "y", type = "double"))
      ),
      flavors = list(crate = list(model = "crate.bin"))
    ),
    file.path(model_dir, "MLmodel")
  )
  saveRDS(list(v = 1), file.path(model_dir, "crate.bin"))

  mock_client <- get_mock_client()
  mock_client$registry_uri <- list(scheme = "databricks-uc")
  mock_client$registry_client <- mock_client

  calls <- list()
  with_mocked_bindings(.package = "mlflow",
    mlflow_materialize_local_model = function(source, client) model_dir,
    mlflow_upload_model_dir_for_uc = function(model_dir, model_version, credentials, client = NULL) {
      calls[[length(calls) + 1]] <<- list(kind = "upload", version = model_version$version)
      TRUE
    },
    mlflow_registry_rest = function(...) {
      args <- list(...)
      calls[[length(calls) + 1]] <<- args
      endpoint <- paste(args[1:2], collapse = "/")
      if (identical(endpoint, "model-versions/create")) {
        return(list(model_version = list(name = "zacdav.default.m", version = "3")))
      }
      if (identical(endpoint, "model-versions/generate-temporary-credentials")) {
        return(list(credentials = list(storage_mode = "DEFAULT_STORAGE")))
      }
      if (identical(endpoint, "model-versions/finalize")) {
        return(list(model_version = list(name = "zacdav.default.m", version = "3")))
      }
      stop("unexpected endpoint")
    }, {
      result <- mlflow_create_model_version(
        name = "zacdav.default.m",
        source = model_dir,
        run_id = "rid-1",
        client = mock_client
      )
      expect_equal(result$version, "3")
    })

  expect_equal(paste(calls[[1]][1:2], collapse = "/"), "model-versions/create")
  expect_equal(paste(calls[[2]][1:2], collapse = "/"), "model-versions/generate-temporary-credentials")
  expect_equal(calls[[3]]$kind, "upload")
  expect_equal(paste(calls[[4]][1:2], collapse = "/"), "model-versions/finalize")
})

test_that("UC create model version delegates customer-managed storage to Python", {
  model_dir <- tempfile("uc-customer-managed-")
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(model_dir, recursive = TRUE), add = TRUE)

  yaml::write_yaml(
    list(
      signature = list(outputs = list(list(name = "y", type = "double"))),
      flavors = list(crate = list(model = "crate.bin"))
    ),
    file.path(model_dir, "MLmodel")
  )
  saveRDS(list(v = 1), file.path(model_dir, "crate.bin"))

  mock_client <- get_mock_client()
  mock_client$registry_uri <- list(scheme = "databricks-uc")
  mock_client$registry_client <- mock_client

  calls <- list()
  python_called <- FALSE
  with_mocked_bindings(.package = "mlflow",
    mlflow_materialize_local_model = function(source, client) model_dir,
    mlflow_uc_sdk_models_artifact_repository_enabled = function(client) FALSE,
    mlflow_upload_and_finalize_uc_model_version_with_python = function(model_dir, model_version,
                                                                       client) {
      python_called <<- TRUE
      expect_equal(model_version$name, "zacdav.default.m")
      expect_equal(model_version$version, "5")
      list(name = model_version$name, version = model_version$version, status = "READY")
    },
    mlflow_registry_rest = function(...) {
      args <- list(...)
      calls[[length(calls) + 1]] <<- args
      endpoint <- paste(args[1:2], collapse = "/")
      if (identical(endpoint, "model-versions/create")) {
        return(list(model_version = list(name = "zacdav.default.m", version = "5")))
      }
      if (identical(endpoint, "model-versions/generate-temporary-credentials")) {
        return(list(credentials = list(storage_mode = "CUSTOMER_HOSTED")))
      }
      if (identical(endpoint, "model-versions/finalize")) {
        stop("unexpected R finalize")
      }
      stop("unexpected endpoint")
    }, {
      result <- mlflow_create_model_version(
        name = "zacdav.default.m",
        source = model_dir,
        client = mock_client
      )
      expect_equal(result$status, "READY")
    })

  expect_true(python_called)
  expect_equal(length(calls), 2)
})

test_that("UC default storage upload requests Databricks upload URLs per file", {
  model_dir <- tempfile("uc-default-storage-")
  dir.create(file.path(model_dir, "nested"), recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(model_dir, recursive = TRUE), add = TRUE)

  writeLines("flavors: {}", file.path(model_dir, "MLmodel"))
  writeLines("payload", file.path(model_dir, "nested", "crate.bin"))

  mock_client <- get_mock_client()
  model_version <- list(name = "zacdav.default.m", version = "4")
  upload_urls <- character()
  uploaded_files <- character()

  with_mocked_bindings(.package = "mlflow",
    mlflow_uc_create_upload_url = function(client, path) {
      upload_urls <<- c(upload_urls, path)
      list(url = paste0("https://signed.example.com/", basename(path)), headers = list())
    },
    mlflow_upload_file_to_signed_url = function(url, file, headers = list()) {
      uploaded_files <<- c(uploaded_files, basename(file))
      invisible(TRUE)
    }, {
      mlflow_upload_model_dir_for_uc(
        model_dir = model_dir,
        model_version = model_version,
        credentials = list(storage_mode = "DEFAULT_STORAGE"),
        client = mock_client
      )
    })

  expect_true("/Models/zacdav/default/m/4/MLmodel" %in% upload_urls)
  expect_true("/Models/zacdav/default/m/4/nested/crate.bin" %in% upload_urls)
  expect_equal(sort(uploaded_files), c("crate.bin", "MLmodel"))
})

test_that("UC SDK models artifact repository uses Databricks file URLs for upload", {
  model_dir <- tempfile("uc-sdk-models-storage-")
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(model_dir, recursive = TRUE), add = TRUE)

  writeLines("flavors: {}", file.path(model_dir, "MLmodel"))

  mock_client <- get_mock_client()
  called <- FALSE

  with_mocked_bindings(.package = "mlflow",
    mlflow_uc_sdk_models_artifact_repository_enabled = function(client) TRUE,
    mlflow_upload_model_dir_to_uc_default_storage = function(model_dir, model_version, client, files) {
      called <<- TRUE
      expect_equal(model_version$name, "zacdav.default.m")
      expect_equal(model_version$version, "4")
      expect_equal(basename(files), "MLmodel")
      invisible(TRUE)
    }, {
      mlflow_upload_model_dir_for_uc(
        model_dir = model_dir,
        model_version = list(name = "zacdav.default.m", version = "4"),
        credentials = list(
          aws_temp_credentials = list(
            access_key_id = "access-key",
            secret_access_key = "secret-key",
            session_token = "session-token"
          )
        ),
        client = mock_client
      )
    })

  expect_true(called)
})

test_that("UC default storage listing recurses and paginates", {
  calls <- list()
  root <- "/Models/zacdav/default/m/4"

  with_mocked_bindings(.package = "mlflow",
    mlflow_uc_list_default_storage_dir = function(client, path, page_token = NULL) {
      calls[[length(calls) + 1]] <<- list(path = path, page_token = page_token)
      if (identical(path, root) && is.null(page_token)) {
        return(list(
          contents = list(
            list(path = paste0(root, "/nested/"), is_directory = TRUE),
            list(path = paste0(root, "/MLmodel"), is_directory = FALSE)
          ),
          next_page_token = "next"
        ))
      }
      if (identical(path, paste0(root, "/nested"))) {
        return(list(
          contents = list(list(path = paste0(root, "/nested/crate.bin"), is_directory = FALSE))
        ))
      }
      if (identical(path, root) && identical(page_token, "next")) {
        return(list(
          contents = list(list(path = paste0(root, "/conda.yaml"), is_directory = FALSE))
        ))
      }
      stop("unexpected list call")
    }, {
      files <- mlflow_uc_default_storage_files(get_mock_client(), root)
    })

  expect_equal(sort(files), sort(c(
    paste0(root, "/MLmodel"),
    paste0(root, "/nested/crate.bin"),
    paste0(root, "/conda.yaml")
  )))
  expect_equal(calls[[3]]$page_token, "next")
})

test_that("UC default storage listing sends page token in GET body", {
  with_mocked_bindings(.package = "mlflow",
    mlflow_rest = function(...) {
      args <- list(...)
      expect_equal(paste(args[1:3], collapse = "/"), "fs/directories/Models/zacdav/default/m/4")
      expect_equal(args$verb, "GET")
      expect_equal(args$path_prefix, "api/2.0")
      expect_null(args$query)
      expect_equal(args$data$page_token, "next")
      list(contents = list())
    }, {
      mlflow_uc_list_default_storage_dir(
        client = get_mock_client(),
        path = "/Models/zacdav/default/m/4",
        page_token = "next"
      )
    })
})

test_that("UC default storage download requests Databricks download URLs per file", {
  mock_client <- get_mock_client()
  model_version <- list(name = "zacdav.default.m", version = "4")
  downloaded_files <- character()
  requested_urls <- character()

  with_mocked_bindings(.package = "mlflow",
    mlflow_uc_default_storage_files = function(client, root) {
      expect_identical(client, mock_client)
      expect_equal(root, "/Models/zacdav/default/m/4")
      c(
        "/Models/zacdav/default/m/4/MLmodel",
        "/Models/zacdav/default/m/4/nested/crate.bin"
      )
    },
    mlflow_uc_create_download_url = function(client, path) {
      requested_urls <<- c(requested_urls, path)
      list(url = paste0("https://signed.example.com/", basename(path)), headers = list())
    },
    mlflow_download_file_from_signed_url = function(url, local_file, headers = list()) {
      downloaded_files <<- c(downloaded_files, local_file)
      dir.create(dirname(local_file), recursive = TRUE, showWarnings = FALSE)
      writeLines("payload", local_file)
      local_file
    }, {
      out <- mlflow_download_model_dir_from_uc_default_storage(model_version, client = mock_client)
      expect_true(file.exists(file.path(out, "MLmodel")))
      expect_true(file.exists(file.path(out, "nested", "crate.bin")))
    })

  expect_equal(requested_urls, c(
    "/Models/zacdav/default/m/4/MLmodel",
    "/Models/zacdav/default/m/4/nested/crate.bin"
  ))
  expect_true(any(grepl("nested/crate\\.bin$", gsub("\\\\", "/", downloaded_files))))
})

test_that("UC model version download uses default storage helper when credentials require it", {
  mock_client <- get_mock_client()
  called <- FALSE

  with_mocked_bindings(.package = "mlflow",
    mlflow_get_model_version = function(name, version, client = NULL) {
      list(name = name, version = version)
    },
    mlflow_uc_model_version_credentials = function(name, version, operation, client) {
      list(storage_mode = "DEFAULT_STORAGE")
    },
    mlflow_get_model_version_download_uri = function(...) stop("unexpected download URI call"),
    mlflow_download_model_dir_from_uc_default_storage = function(model_version, client) {
      called <<- TRUE
      "downloaded-model"
    }, {
      expect_equal(
        mlflow_download_uc_model_version("zacdav.default.m", "4", client = mock_client),
        "downloaded-model"
      )
    })

  expect_true(called)
})

test_that("UC SDK models artifact repository uses Databricks file URLs for download", {
  mock_client <- get_mock_client()
  called <- FALSE

  with_mocked_bindings(.package = "mlflow",
    mlflow_get_model_version = function(name, version, client = NULL) {
      list(name = name, version = version, storage_location = "s3://bucket/path/to/model")
    },
    mlflow_uc_model_version_credentials = function(name, version, operation, client) {
      list(
        aws_temp_credentials = list(
          access_key_id = "access-key",
          secret_access_key = "secret-key",
          session_token = "session-token"
        )
      )
    },
    mlflow_uc_sdk_models_artifact_repository_enabled = function(client) TRUE,
    mlflow_get_model_version_download_uri = function(...) stop("unexpected download URI call"),
    mlflow_download_model_dir_from_uc_default_storage = function(model_version, client) {
      called <<- TRUE
      "downloaded-model"
    }, {
      expect_equal(
        mlflow_download_uc_model_version("zacdav.default.m", "4", client = mock_client),
        "downloaded-model"
      )
    })

  expect_true(called)
})

test_that("UC SDK models artifact repository feature flag is read from registry endpoint", {
  with_mocked_bindings(.package = "mlflow",
    mlflow_registry_rest = function(...) {
      args <- list(...)
      expect_equal(
        args[[1]],
        "registered-models:is-databricks-sdk-models-artifact-repository-enabled"
      )
      expect_equal(args$verb, "GET")
      list(is_databricks_sdk_models_artifact_repository_enabled = TRUE)
    }, {
      expect_true(mlflow_uc_sdk_models_artifact_repository_enabled(get_mock_client()))
    })
})

test_that("UC unsupported temporary credentials fail before finalization", {
  model_dir <- tempfile("uc-unsupported-credentials-")
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(model_dir, recursive = TRUE), add = TRUE)

  writeLines("flavors: {}", file.path(model_dir, "MLmodel"))

  with_mocked_bindings(.package = "mlflow",
    mlflow_uc_sdk_models_artifact_repository_enabled = function(client) FALSE, {
      expect_error(
        mlflow_upload_model_dir_for_uc(
          model_dir = model_dir,
          model_version = list(name = "zacdav.default.m", version = "4"),
          credentials = list(gcp_oauth_token = list(oauth_token = "x")),
          client = get_mock_client()
        ),
        "non-default model-version storage credentials",
        fixed = TRUE
      )
    })
})

test_that("UC customer-managed download delegates to Python", {
  mock_client <- get_mock_client()
  python_called <- FALSE
  with_mocked_bindings(.package = "mlflow",
    mlflow_get_model_version = function(name, version, client = NULL) {
      list(name = name, version = version, storage_location = "s3://bucket/path/to/model")
    },
    mlflow_uc_model_version_credentials = function(name, version, operation, client) {
      list(
        aws_temp_credentials = list(
          access_key_id = "access-key",
          secret_access_key = "secret-key",
          session_token = "session-token"
        )
      )
    },
    mlflow_uc_sdk_models_artifact_repository_enabled = function(client) FALSE,
    mlflow_download_artifacts_from_uri = function(...) stop("unexpected artifact download"),
    mlflow_download_uc_model_version_with_python = function(name, version, client) {
      python_called <<- TRUE
      expect_equal(name, "zacdav.default.m")
      expect_equal(version, "4")
      "downloaded-model"
    }, {
      expect_equal(
        mlflow_download_uc_model_version("zacdav.default.m", "4", client = mock_client),
        "downloaded-model"
      )
    })

  expect_true(python_called)
})

test_that("UC customer-managed download does not use generic download URI lookup", {
  mock_client <- get_mock_client()
  with_mocked_bindings(.package = "mlflow",
    mlflow_get_model_version = function(name, version, client = NULL) {
      list(name = name, version = version)
    },
    mlflow_uc_model_version_credentials = function(name, version, operation, client) {
      list(gcp_oauth_token = list(oauth_token = "x"))
    },
    mlflow_uc_sdk_models_artifact_repository_enabled = function(client) FALSE,
    mlflow_get_model_version_download_uri = function(...) stop("unexpected download URI lookup"),
    mlflow_download_artifacts_from_uri = function(...) stop("unexpected artifact download"),
    mlflow_download_uc_model_version_with_python = function(name, version, client) {
      "downloaded-model"
    }, {
      expect_equal(
        mlflow_download_uc_model_version("zacdav.default.m", "4", client = mock_client),
        "downloaded-model"
      )
    })
})

test_that("UC create model version fails fast when signature is missing", {
  model_dir <- file.path(tempdir(), paste0("uc-no-signature-", as.integer(Sys.time())))
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(model_dir, recursive = TRUE), add = TRUE)
  yaml::write_yaml(list(flavors = list(crate = list(model = "crate.bin"))), file.path(model_dir, "MLmodel"))
  saveRDS(list(v = 1), file.path(model_dir, "crate.bin"))

  mock_client <- get_mock_client()
  mock_client$registry_uri <- list(scheme = "databricks-uc")
  mock_client$registry_client <- mock_client

  with_mocked_bindings(.package = "mlflow",
    mlflow_materialize_local_model = function(source, client) model_dir, {
      expect_error(
        mlflow_create_model_version(name = "zacdav.default.m", source = model_dir, client = mock_client),
        "must include a model signature"
      )
  })
})

test_that("UC signature validation requires output type specifications", {
  model_dir <- tempfile("uc-signature-")
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(model_dir, recursive = TRUE), add = TRUE)

  yaml::write_yaml(
    list(
      signature = list(outputs = list(list(name = "prediction", type = "double"))),
      flavors = list(crate = list(model = "crate.bin"))
    ),
    file.path(model_dir, "MLmodel")
  )
  expect_true(mlflow_validate_uc_model_signature(model_dir))

  yaml::write_yaml(
    list(
      signature = list(inputs = list(list(name = "feature", type = "double"))),
      flavors = list(crate = list(model = "crate.bin"))
    ),
    file.path(model_dir, "MLmodel")
  )
  expect_error(
    mlflow_validate_uc_model_signature(model_dir),
    "output type specifications",
    fixed = TRUE
  )

  yaml::write_yaml(
    list(
      signature = list(outputs = ""),
      flavors = list(crate = list(model = "crate.bin"))
    ),
    file.path(model_dir, "MLmodel")
  )
  expect_error(
    mlflow_validate_uc_model_signature(model_dir),
    "output type specifications",
    fixed = TRUE
  )
})
