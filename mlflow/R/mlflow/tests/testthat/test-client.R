context("databricks-utils")

test_that("http(s) clients work as expected", {
  with_mock(.env = "mlflow", mlflow_rest = function(..., client) {
    args <- list(...)
    expect_true(paste(args, collapse = "/") == "experiments/list")
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

test_that("rest call handles errors correctly", {
  with_mock(.env = "httr", GET = function(...) {
    httr:::response(
      status_code = 400,
      content = charToRaw(paste("{\"error_code\":\"INVALID_PARAMETER_VALUE\",",
                                 "\"message\":\"experiment_id must be set to a non-zero value\"}",
                                 sep = "")
      )
    )
    expected_error_msg <- paste(
      "Error in mlflow_rest(\"runs\", \"create\", client = client_db, verb = \"POST\",  :",
      "API request to endpoint 'runs/create' failed with error code 400. Reponse body: ",
      "'INVALID_PARAMETER_VALUE, experiment_id must be set to a non-zero value'",
      , sep = ""
    )
    client <- mlflow:::mlflow_client()
    expect_error(
      mlflow:::mlflow_rest(mlflow:::mlflow_rest( "runs", "create", client = client, verb = "POST")),
      expected_error_msg
    )
  })
})
