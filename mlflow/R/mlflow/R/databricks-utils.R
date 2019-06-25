# Utils for databricks authentication

new_mlflow_client.mlflow_databricks <- function(tracking_uri) {
  profile <- tracking_uri$path
  # make sure we can read the config
  new_mlflow_client_impl(
    get_host_creds = function() {
      get_databricks_config(profile)
    },
    get_cli_env = function() {
      databricks_config_as_env(get_databricks_config(profile))
    },
    class = "mlflow_databricks_client"
  )
}

DATABRICKS_CONFIG_FILE <- "DATABRICKS_CONFIG_FILE"
# map expected config variables to environment variables
config_variable_map <- list(
  host = "DATABRICKS_HOST",
  username = "DATABRICKS_USERNAME",
  password = "DATABRICKS_PASSWORD",
  token = "DATABRICKS_TOKEN",
  insecure = "DATABRICKS_INSECURE"
)
  
databricks_config_as_env <- function(config) {
  if (config$config_source != "cfgfile") { # pass the auth info via environment vars
    res <- config[!is.na(config)]
    res$config_source <- NULL
    if (!as.logical(res$insecure)) {
      res$insecure <- NULL
    }
    names(res) <- lapply(names(res), function (x) config_variable_map[[x]])
    res
  } else if (!is.na(Sys.getenv(DATABRICKS_CONFIG_FILE, NA))) {
    list(DATABRICKS_CONFIG_FILE = Sys.getenv(DATABRICKS_CONFIG_FILE))
  } else {
    # We do not need to do anything if the config comes from a file visible to both processes
    list()
  }
}

databricks_config_is_valid <- function(config) {
  !is.na(config$host) &&
      (!is.na(config$token) || (!is.na(config$username) && !is.na(config$password)))
}

#' @importFrom ini read.ini
get_databricks_config_for_profile <- function(profile) {
  config_path <- Sys.getenv("DATABRICKS_CONFIG_FILE", NA)
  config_path <- if (is.na(config_path)) path.expand("~/.databrickscfg") else config_path
  if (!file.exists(config_path)){
    stop(paste("Databricks configuration file is missing. Expected config file ", config_path))
  }
  config <- read.ini(config_path)
  if (!(profile %in% names(config))) {
    stop(paste("Missing profile '", profile, "'.", sep = ""))
  }
  new_databricks_config(config_source = "cfgfile", config[[profile]])
}

#' @importFrom utils modifyList
new_databricks_config <- function(config_source,
                                  config_vars) {
  res <- do.call(new_mlflow_host_creds, config_vars)
  res$config_source <- config_source
  res
}

get_databricks_config_from_env <- function() {
  config_vars <- lapply(config_variable_map, function(x) Sys.getenv(x, NA))
  names(config_vars) <- names(config_variable_map)
  new_databricks_config("env", config_vars)
}

get_databricks_config <- function(profile) {
  config <- if (!is.na(profile)) {
    get_databricks_config_for_profile(profile)
  } else if (exists("spark.databricks.token") && exists("spark.databricks.api.url")) {
    config_vars <- list(
      host = get("spark.databricks.api.url", envir = .GlobalEnv),
      token = get("spark.databricks.token", envir = .GlobalEnv),
      insecure = Sys.getenv(config_variable_map$insecure, "False")
    )
    new_databricks_config(config_source = "db_dynamic", config_vars = config_vars)
  } else {
    config <- get_databricks_config_from_env()
    if (databricks_config_is_valid(config)) {
      config
    } else {
      get_databricks_config_for_profile("DEFAULT")
    }
  }
  if (!databricks_config_is_valid(config)) {
    stop("Could not find valid Databricks configuration.")
  }
  config
}

mlflow_get_run_context.mlflow_databricks_client <- function(client, experiment_id, ...) {
  if (exists(".databricks_internals")) {
    notebook_info <- do.call(".get_notebook_info", list(), envir = get(".databricks_internals",
                                                                       envir = .GlobalEnv))
    if (!is.na(notebook_info$id) && !is.na(notebook_info$path)) {
      tags <- list()
      tags[[MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_NOTEBOOK_ID]] <- notebook_info$id
      tags[[MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_NOTEBOOK_PATH]] <- notebook_info$path
      tags[[MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_WEBAPP_URL]] <- notebook_info$webapp_url
      tags[[MLFLOW_TAGS$MLFLOW_SOURCE_NAME]] <- notebook_info$path
      tags[[MLFLOW_TAGS$MLFLOW_SOURCE_VERSION]] <- get_source_version()
      tags[[MLFLOW_TAGS$MLFLOW_SOURCE_TYPE]] <- MLFLOW_SOURCE_TYPE$NOTEBOOK
      list(
        client = client,
        tags = tags,
        experiment_id = experiment_id %||% notebook_info$id,
        ...
      )
    } else {
      NextMethod()
    }
  } else {
    NextMethod()
  }
}

MLFLOW_DATABRICKS_TAGS <- list(
  MLFLOW_DATABRICKS_NOTEBOOK_ID = "mlflow.databricks.notebookID",
  MLFLOW_DATABRICKS_NOTEBOOK_PATH = "mlflow.databricks.notebookPath",
  MLFLOW_DATABRICKS_WEBAPP_URL = "mlflow.databricks.webappURL",
  MLFLOW_DATABRICKS_RUN_URL = "mlflow.databricks.runURL",
  MLFLOW_DATABRICKS_SHELL_JOB_ID = "mlflow.databricks.shellJobID",
  MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID = "mlflow.databricks.shellJobRunID"
)
