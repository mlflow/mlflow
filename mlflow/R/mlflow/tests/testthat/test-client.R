context("client")

teardown({
  mlflow_clear_test_dir("mlruns")
})

test_that("http(s) clients work as expected", {
  mlflow_clear_test_dir("mlruns")
  with_mocked_bindings(.package = "mlflow", mlflow_rest = function(..., client) {
    args <- list(...)
    expect_true(paste(args[1:2], collapse = "/") == "experiments/search")
    list(experiments = c(1, 2, 3))
  }, {
    with_mocked_bindings(.package = "mlflow", mlflow_register_local_server = function(...) NA, {
      env <- list(
        MLFLOW_TRACKING_USERNAME = "DonaldDuck",
        MLFLOW_TRACKING_PASSWORD = "Quack",
        MLFLOW_TRACKING_TOKEN = "$$$",
        MLFLOW_TRACKING_INSECURE = "True"
      )
      with_envvar(env, {
        http_host <- "http://remote"
        client1 <- mlflow:::mlflow_client(http_host)
        config <- client1$get_host_creds()
        print(config)
        expect_true(config$host == http_host)
        expect_true(config$username == "DonaldDuck")
        expect_true(config$password == "Quack")
        expect_true(config$token == "$$$")
        expect_true(config$insecure == "True")
        https_host <- "https://remote"
        client2 <- mlflow:::mlflow_client("https://remote")
        config <- client2$get_host_creds()
        expect_true(config$host == https_host)
        env_str <- paste(env, collapse = "|")
        env_str_2 <- paste(client2$get_cli_env(), collapse = "|")
        expect_true(env_str == env_str_2)
      })
      with_mocked_bindings(.package = "mlflow", mlflow_server = function(...) list(server_url = "local_server"), {
        client3 <- mlflow:::mlflow_client()
        config <- client3$get_host_creds()
        expect_true(config$host == "local_server")
      })
    })
  })
})


test_that("http(s) clients works with deprecated env vars", {
  mlflow_clear_test_dir("mlruns")
  with_mocked_bindings(.package = "mlflow", mlflow_rest = function(..., client) {
    args <- list(...)
    expect_true(paste(args[1:2], collapse = "/") == "experiments/search")
    list(experiments = c(1, 2, 3))
  }, {
    with_mocked_bindings(.package = "mlflow", mlflow_register_local_server = function(...) NA, {
      env <- list(
        MLFLOW_USERNAME = "DonaldDuck",
        MLFLOW_PASSWORD = "Quack",
        MLFLOW_TOKEN = "$$$",
        MLFLOW_INSECURE = "True"
      )
      with_envvar(env, {
        http_host <- "http://remote"
        client1 <- mlflow:::mlflow_client(http_host)
        config <- client1$get_host_creds()
        print(config)
        expect_true(config$host == http_host)
        expect_true(config$username == "DonaldDuck")
        expect_true(config$password == "Quack")
        expect_true(config$token == "$$$")
        expect_true(config$insecure == "True")
        https_host <- "https://remote"
        client2 <- mlflow:::mlflow_client("https://remote")
        config <- client2$get_host_creds()
        expect_true(config$host == https_host)
        env_str <- paste(list(
          MLFLOW_TRACKING_USERNAME = "DonaldDuck",
          MLFLOW_TRACKING_PASSWORD = "Quack",
          MLFLOW_TRACKING_TOKEN = "$$$",
          MLFLOW_TRACKING_INSECURE = "True"
        ), collapse = "|")
        env_str_2 <- paste(client2$get_cli_env(), collapse = "|")
        expect_true(env_str == env_str_2)
      })

      with_mocked_bindings(.package = "mlflow", mlflow_server = function(...) list(server_url = "local_server"), {
        client3 <- mlflow:::mlflow_client()
        config <- client3$get_host_creds()
        expect_true(config$host == "local_server")
      })
    })
  })
})

test_that("rest call handles errors correctly", {
  mlflow_clear_test_dir("mlruns")
  mock_client <- mlflow:::new_mlflow_client_impl(get_host_creds = function() {
     mlflow:::new_mlflow_host_creds(host = "localhost")
  })
  with_mocked_bindings(.package  = "httr", POST = function(...) {
    httr:::response(
      status_code = 400,
      content = charToRaw(paste("{\"error_code\":\"INVALID_PARAMETER_VALUE\",",
                                 "\"message\":\"experiment_id must be set to a non-zero value\"}",
                                 sep = "")
      )
    )}, {
    error_msg_regexp <- paste(
                          "API request to endpoint \'runs/create\' failed with error code 400",
                          "INVALID_PARAMETER_VALUE",
                          "experiment_id must be set to a non-zero value",
                          sep = ".*")
    expect_error(
      mlflow:::mlflow_rest( "runs", "create", client = mock_client, verb = "POST"),
      error_msg_regexp
    )
  })

  with_mocked_bindings(.package  = "httr", GET = function(...) {
    httr:::response(
      status_code = 500,
      content = charToRaw(paste("some text."))
    )
    }, {
    error_msg_regexp <- paste(
                          "API request to endpoint \'runs/create\' failed with error code 500",
                          "some text",
                          sep = ".*")
    expect_error(
      mlflow:::mlflow_rest( "runs", "create", client = mock_client, verb = "GET"),
      error_msg_regexp
    )
  })

  with_mocked_bindings(.package  = "httr", POST = function(...) {
    httr:::response(
      status_code = 503,
      content = as.raw(c(0, 255))
    )
    }, {
    error_msg_regexp <- paste(
                          "API request to endpoint \'runs/create\' failed with error code 503",
                          "00 ff",
                          sep = ".*")
    expect_error(
      mlflow:::mlflow_rest( "runs", "create", client = mock_client, verb = "POST"),
      error_msg_regexp
    )
  })
})

test_that("registry URI defaults and setters work", {
  mlflow_clear_test_dir("mlruns")
  old_tracking <- mlflow_get_tracking_uri()
  old_registry <- .globals$registry_uri
  on.exit({
    mlflow_set_tracking_uri(old_tracking)
    .globals$registry_uri <- old_registry
  }, add = TRUE)

  mlflow_set_tracking_uri("databricks")
  .globals$registry_uri <- NULL
  expect_equal(mlflow_get_registry_uri(), "databricks-uc")

  mlflow_set_tracking_uri("databricks://PROFILE")
  .globals$registry_uri <- NULL
  expect_equal(mlflow_get_registry_uri(), "databricks-uc://PROFILE")

  client <- mlflow_client("databricks://PROFILE")
  expect_equal(client$registry_uri$scheme, "databricks-uc")
  expect_equal(client$registry_uri$path, "PROFILE")

  client <- mlflow_client("databricks")
  expect_equal(client$tracking_uri$raw_uri, "databricks")
  expect_equal(client$registry_uri$raw_uri, "databricks-uc")

  with_envvar(c(MLFLOW_REGISTRY_URI = "databricks://PROD"), {
    .globals$registry_uri <- NULL
    expect_equal(mlflow_get_registry_uri(), "databricks://PROD")
  })

  mlflow_set_registry_uri("databricks://DEFAULT")
  expect_equal(mlflow_get_registry_uri(), "databricks://DEFAULT")

  mlflow_set_registry_uri("")
  mlflow_set_tracking_uri("databricks")
  expect_equal(mlflow_get_registry_uri(), "databricks-uc")

  with_envvar(c(MLFLOW_REGISTRY_URI = "databricks://PROD"), {
    mlflow_set_registry_uri("")
    expect_equal(mlflow_get_registry_uri(), "databricks-uc")
  })

  mlflow_set_tracking_uri("http://localhost:5000")
  .globals$registry_uri <- NULL
  expect_equal(mlflow_get_registry_uri(), "http://localhost:5000")
})

test_that("mlflow_cli passes registry URI env", {
  old_tracking <- mlflow_get_tracking_uri()
  old_registry <- .globals$registry_uri
  on.exit({
    mlflow_set_tracking_uri(old_tracking)
    .globals$registry_uri <- old_registry
  }, add = TRUE)

  mlflow_set_tracking_uri("http://tracking")
  mlflow_set_registry_uri("http://registry")

  env <- NULL
  with_mocked_bindings(.package = "mlflow",
    python_bin = function() "/usr/bin/python",
    python_mlflow_bin = function() "/usr/bin/mlflow",
    run = function(command = NULL, args = character(), echo = TRUE, echo_cmd = FALSE,
                   stderr_callback = NULL) {
      env <<- list(
        tracking_uri = Sys.getenv("MLFLOW_TRACKING_URI"),
        registry_uri = Sys.getenv("MLFLOW_REGISTRY_URI")
      )
      list(stdout = "")
    }, {
      mlflow_cli("experiments", "list", client = NULL)
    })

  expect_equal(env$tracking_uri, "http://tracking")
  expect_equal(env$registry_uri, "http://registry")
})

test_that("mlflow_cli uses client tracking and registry URIs", {
  mock_client <- new_mlflow_client_impl(get_host_creds = function() {
    new_mlflow_host_creds(host = "localhost")
  })
  mock_client$tracking_uri <- list(raw_uri = "databricks://PROFILE")
  mock_client$registry_uri <- list(raw_uri = "databricks-uc://PROFILE")

  env <- NULL
  with_mocked_bindings(.package = "mlflow",
    python_bin = function() "/usr/bin/python",
    python_mlflow_bin = function() "/usr/bin/mlflow",
    run = function(command = NULL, args = character(), echo = TRUE, echo_cmd = FALSE,
                   stderr_callback = NULL) {
      env <<- list(
        tracking_uri = Sys.getenv("MLFLOW_TRACKING_URI"),
        registry_uri = Sys.getenv("MLFLOW_REGISTRY_URI")
      )
      list(stdout = "")
    }, {
      mlflow_cli("experiments", "list", client = mock_client)
    })

  expect_equal(env$tracking_uri, "databricks://PROFILE")
  expect_equal(env$registry_uri, "databricks-uc://PROFILE")
})
