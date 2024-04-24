context("client")

teardown({
  mlflow_clear_test_dir("mlruns")
})

test_that("http(s) clients work as expected", {
  mlflow_clear_test_dir("mlruns")
  with_mock(.env = "mlflow", mlflow_rest = function(..., client) {
    args <- list(...)
    expect_true(paste(args[1:2], collapse = "/") == "experiments/search")
    list(experiments = c(1, 2, 3))
  }, {
    with_mock(.env = "mlflow", mlflow_register_local_server = function(...) NA, {
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
      with_mock(.env = "mlflow", mlflow_server = function(...) list(server_url = "local_server"), {
        client3 <- mlflow:::mlflow_client()
        config <- client3$get_host_creds()
        expect_true(config$host == "local_server")
      })
    })
  })
})


test_that("http(s) clients works with deprecated env vars", {
  mlflow_clear_test_dir("mlruns")
  with_mock(.env = "mlflow", mlflow_rest = function(..., client) {
    args <- list(...)
    expect_true(paste(args[1:2], collapse = "/") == "experiments/search")
    list(experiments = c(1, 2, 3))
  }, {
    with_mock(.env = "mlflow", mlflow_register_local_server = function(...) NA, {
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

      with_mock(.env = "mlflow", mlflow_server = function(...) list(server_url = "local_server"), {
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
  with_mock(.env = "httr", POST = function(...) {
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

  with_mock(.env = "httr", GET = function(...) {
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

  with_mock(.env = "httr", POST = function(...) {
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
