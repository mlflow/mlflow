context("databricks-utils")

library(withr)

test_that("mlflow creates databricks client when scheme is databricks", {
  with_mock(.env = "mlflow", get_databricks_config = function(profile) {
      config_vars <- list(host = "databricks-host", token = "databricks")
      config <- new_databricks_config( config_source = "env", config_vars = config_vars)
      config$profile <- profile
      config
  }, {
    with_mock(.env = "mlflow", mlflow_rest = function(..., client) {
      args <- list(...)
      expect_true(paste(args, collapse = "/") == "experiments/list")
      list(experiments = c(1, 2, 3))
    }, {
      client1 <- mlflow:::mlflow_client("databricks")
      creds1 <- client1$get_host_creds()
      expect_true(creds1$host == "databricks-host")
      expect_true(creds1$token == "databricks")
      expect_true(is.na(creds1$profile))
      env1 <- client1$get_cli_env()
      expect_true(env1$DATABRICKS_HOST == "databricks-host")
      expect_true(env1$DATABRICKS_TOKEN == "databricks")
      client2 <- mlflow:::mlflow_client("databricks://dbprofile")
      creds2 <- client2$get_host_creds()
      expect_true(creds2$host == "databricks-host")
      expect_true(creds2$token == "databricks")
      expect_true(creds2$profile == "dbprofile")
    })
  })
})

test_that("mlflow reads databricks config from correct sources", {
  with_mock(.env = "mlflow", get_databricks_config_for_profile = function(profile) list(
    host = "databricks-host", token = "databricks", profile = profile), {
      config <- get_databricks_config("profile")
      expect_true(config$profile == "profile")
      expect_true(config$host == "databricks-host")
      expect_true(config$host == "databricks-host")
      expect_true(config$token == "databricks")
      config <- get_databricks_config(NA)
      expect_true(config$profile == "DEFAULT")
      expect_true(config$host == "databricks-host")
      expect_true(config$host == "databricks-host")
      expect_true(config$token == "databricks")

      with_mock(.env = "mlflow",
        get_databricks_config_from_env = function() {
            new_databricks_config("env", list(host = "host"))
        }, {
        config <- get_databricks_config(NA)
        expect_true(config$profile == "DEFAULT")
        expect_true(config$host == "databricks-host")
        expect_true(config$host == "databricks-host")
        expect_true(config$token == "databricks")
      })
      with_mock(.env = "mlflow",
        get_databricks_config_from_env = function() {
            new_databricks_config("env", list(host = "env", token = "env"))
        }, {
        config <- get_databricks_config(NA)
        expect_true(config$host == "env")
        expect_true(config$token == "env")
      })
  })
})

test_that("mlflow can read .databrickscfg files", {
  config_file <- file.path(tempdir(), ".databrickscfg")
  Sys.setenv(DATABRICKS_CONFIG_FILE = config_file)
   tryCatch(
    {
      config_file <- file.path(tempdir(), ".databrickscfg")
      profile1 <- c("[PROFILE1]", "host = host1", "token = token1")
      donald <- c("[DONALD]", "host = duckburg", "username = donaldduck", "password = quackquack",
                  "insecure = True")
      broken_1 <- c("[BROKEN1]", "username = donaldduck", "token = abc")
      broken_2 <- c("[BROKEN2]", "username = donaldduck", "host = duckburg")
      profiles <- c(profile1, default_profile, donald, broken_1, broken_2)
      write(profiles, file = config_file,
            ncolumns = 1, append = FALSE, sep = "\n")

       profile1 <- mlflow:::get_databricks_config_for_profile("PROFILE1")
       expect_true(profile1$config_source == "cfgfile")
       expect_true(profile1$host == "host1")
       expect_true(profile1$token == "token1")
       expect_true(is.na(profile1$username))
       expect_true(is.na(profile1$password))
       expect_true(profile1$insecure == "False")
       expect_true(mlflow:::databricks_config_is_valid(profile1))

       profile2 <- mlflow:::get_databricks_config_for_profile("DONALD")
       expect_true(profile2$config_source == "cfgfile")
       expect_true(profile2$host == "duckburg")
       expect_true(is.na(profile2$token))
       expect_true(profile2$username == "donaldduck")
       expect_true(profile2$password == "quackquack")
       expect_true(profile2$insecure == "True")
       expect_true(mlflow:::databricks_config_is_valid(profile2))

       profile3 <- mlflow:::get_databricks_config_for_profile("BROKEN1")
       expect_true(profile3$config_source == "cfgfile")
       expect_true(is.na(profile3$host))
       expect_true(profile3$token == "abc")
       expect_true(profile3$username == "donaldduck")
       expect_true(is.na(profile3$password))
       expect_true(profile3$insecure == "False")
       expect_false(mlflow:::databricks_config_is_valid(profile3))

       profile4 <- mlflow:::get_databricks_config_for_profile("BROKEN2")
       expect_true(profile4$config_source == "cfgfile")
       expect_true(profile4$host == "duckburg")
       expect_true(is.na(profile4$token))
       expect_true(profile4$username == "donaldduck")
       expect_true(is.na(profile4$password))
       expect_true(profile1$insecure == "False")
       expect_false(mlflow:::databricks_config_is_valid(profile4))

       unlink(config_file)
       Sys.unsetenv(DATABRICKS_CONFIG_FILE)
    },
    error = function(cnd) {
      unlink(config_file)
      Sys.unsetenv(DATABRICKS_CONFIG_FILE)
    },
    interrupt = function(cnd) {
      unlink(config_file)
      Sys.unsetenv(DATABRICKS_CONFIG_FILE)
    }
  )
})

test_that("mlflow can read databricks env congfig", {
  env <- list(
    DATABRICKS_HOST = "envhost",
    DATABRICKS_USERNAME = "envusername",
    DATABRICKS_PASSWORD = "envpassword",
    DATABRICKS_TOKEN = "envtoken",
    DATABRICKS_INSECURE = "True")
  with_envvar( env, {
      envprofile <- mlflow:::get_databricks_config_from_env()
      expect_true(envprofile$host == "envhost")
      expect_true(envprofile$token == "envtoken")
      expect_true(envprofile$username == "envusername")
      expect_true(envprofile$password == "envpassword")
      expect_true(envprofile$insecure == "True")
      expect_true(mlflow:::databricks_config_is_valid(envprofile))

      extracted_env <- mlflow:::databricks_config_as_env(envprofile)
      expect_true(paste(env, collapse = "|") == paste(extracted_env, collapse = "|"))
      expect_true(length(setdiff(extracted_env, env)) == 0)
    }
  )
  env <- list(DATABRICKS_HOST = "envhost",
              DATABRICKS_USERNAME = "envusername",
              DATABRICKS_TOKEN = "envtoken")

  with_envvar(env, {
    envprofile <- mlflow:::get_databricks_config_from_env()
    expect_true(envprofile$host == "envhost")
    expect_true(envprofile$token == "envtoken")
    expect_true(envprofile$username == "envusername")
    expect_true(is.na(envprofile$password))
    expect_true(envprofile$insecure == "False")
    expect_true(mlflow:::databricks_config_is_valid(envprofile))
    extracted_env <- mlflow:::databricks_config_as_env(envprofile)
    expect_true(paste(env, collapse = "|") == paste(extracted_env, collapse = "|"))
    expect_true(length(setdiff(extracted_env, env)) == 0)
  })
})

#' Executes the specified functions while creating mock `.get_notebook_info` and
#' `.get_job_info` functions in the `.databricks_internals` environment
run_with_mock_notebook_and_job_info <- function(func, notebook_info = NULL, job_info = NULL) {
  tryCatch({
    assign(".databricks_internals", new.env(parent = baseenv()), envir = .GlobalEnv)
    databricks_internal_env = get(".databricks_internals", envir = .GlobalEnv)
    assign(".get_notebook_info", function() {
        if (is.null(notebook_info)) {
          notebook_info <- list(id = NA, path = NA, webapp_url = NA)
        }
        notebook_info
      },
      envir = databricks_internal_env)
    if (!is.null(job_info)) {
      assign(".get_job_info", function() { job_info }, envir = databricks_internal_env)
    }
    return(func())
  },
  finally = {
    rm(".databricks_internals", envir = .GlobalEnv)
  })
}

test_that("databricks get run context fetches expected info for notebook environment", {
  mock_notebook_info <- list(id = "5",
                             path = "/path/to/notebook",
                             webapp_url = "https://databricks-url.com")
  expected_tags = list("5", "/path/to/notebook", "https://databricks-url.com",
                       MLFLOW_SOURCE_TYPE$NOTEBOOK, "/path/to/notebook")
  names(expected_tags) <- c(
    MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_NOTEBOOK_ID,
    MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_NOTEBOOK_PATH,
    MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_WEBAPP_URL,
    MLFLOW_TAGS$MLFLOW_SOURCE_TYPE,
    MLFLOW_TAGS$MLFLOW_SOURCE_NAME
  )

  run_with_mock_notebook_and_job_info(function() {
    context <- mlflow:::mlflow_get_run_context.mlflow_databricks_client(
      client = mlflow:::mlflow_client("databricks"),
      experiment_id = "52"
    )
    expect_true(all(expected_tags %in% context$tags))
    expect_true(context$experiment_id == "52")
  }, notebook_info = mock_notebook_info, job_info = NULL)

  run_with_mock_notebook_and_job_info(function() {
    context <- mlflow:::mlflow_get_run_context.mlflow_databricks_client(
      client = mlflow:::mlflow_client("databricks"),
      experiment_id = NULL
    )
    expect_true(all(expected_tags %in% context$tags))
    expect_true(context$experiment_id == "5")
  }, notebook_info = mock_notebook_info, job_info = NULL)
})

test_that("databricks get run context fetches expected info for job environment", {
  mock_job_info <- list(job_id = "10",
                        run_id = "2",
                        job_type = "notebook",
                        webapp_url = "https://databricks-url.com")
  expected_tags = list("10", "2", "notebook", "https://databricks-url.com",
                       MLFLOW_SOURCE_TYPE$JOB, "jobs/10/run/2")
  names(expected_tags) <- c(
    MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_JOB_ID,
    MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_JOB_RUN_ID,
    MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_JOB_TYPE,
    MLFLOW_TAGS$MLFLOW_SOURCE_TYPE,
    MLFLOW_TAGS$MLFLOW_SOURCE_NAME
  )

  run_with_mock_notebook_and_job_info(function() {
    context <- mlflow:::mlflow_get_run_context.mlflow_databricks_client(
      mlflow:::mlflow_client("databricks"),
      experiment_id = "52"
    )
    expect_true(all(expected_tags %in% context$tags))
    expect_true(context$experiment_id == "52")
  }, notebook_info = NULL, job_info = mock_job_info)

  run_with_mock_notebook_and_job_info(function() {
    context <- mlflow:::mlflow_get_run_context.mlflow_databricks_client(
      mlflow:::mlflow_client("databricks"),
      experiment_id = NULL
    )
    expect_true(all(expected_tags %in% context$tags))
    expect_true(context$experiment_id == 0)
  }, notebook_info = NULL, job_info = mock_job_info)
})

#' Verifies that, when notebook information is unavailable and there is no function
#' available for fetching job info (as is the case for older Spark images), fetching
#' the databricks run context delegates to the next context provider
test_that("databricks get run context succeeds when job info function is unavailable", {
  run_with_mock_notebook_and_job_info(function() {
    test_env = new.env()
    test_env$next_method_calls <- 0
    mock_next_method = function() {
      test_env$next_method_calls <- test_env$next_method_calls + 1
    }
    testthat::with_mock(NextMethod = mock_next_method, {
      context <- mlflow:::mlflow_get_run_context.mlflow_databricks_client(
        mlflow:::mlflow_client("databricks"),
        experiment_id = 0
      )
    })
    expect_true(test_env$next_method_calls == 1)
  }, notebook_info = NULL, job_info = NULL)
})
