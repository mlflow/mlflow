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
