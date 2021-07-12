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

#' @return A list of tags to be set by the run context when creating MLflow runs in the
#' current Databricks Notebook environment
build_context_tags_from_databricks_notebook_info <- function(notebook_info) {
  tags <- list()
  tags[[MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_NOTEBOOK_ID]] <- notebook_info$id
  tags[[MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_NOTEBOOK_PATH]] <- notebook_info$path
  tags[[MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_WEBAPP_URL]] <- notebook_info$webapp_url
  tags[[MLFLOW_TAGS$MLFLOW_SOURCE_NAME]] <- notebook_info$path
  tags[[MLFLOW_TAGS$MLFLOW_SOURCE_VERSION]] <- get_source_version()
  tags[[MLFLOW_TAGS$MLFLOW_SOURCE_TYPE]] <- MLFLOW_SOURCE_TYPE$NOTEBOOK
  tags
}

#' @return A list of tags to be set by the run context when creating MLflow runs in the
#' current Databricks Job environment
build_context_tags_from_databricks_job_info <- function(job_info) {
  tags <- list()
  tags[[MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_JOB_ID]] <- job_info$job_id
  tags[[MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_JOB_RUN_ID]] <- job_info$run_id
  tags[[MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_JOB_TYPE]] <- job_info$job_type
  tags[[MLFLOW_DATABRICKS_TAGS$MLFLOW_DATABRICKS_WEBAPP_URL]] <- job_info$webapp_url
  tags[[MLFLOW_TAGS$MLFLOW_SOURCE_NAME]] <- paste(
    "jobs", job_info$job_id, "run", job_info$run_id, sep = "/"
  )
  tags[[MLFLOW_TAGS$MLFLOW_SOURCE_VERSION]] <- get_source_version()
  tags[[MLFLOW_TAGS$MLFLOW_SOURCE_TYPE]] <- MLFLOW_SOURCE_TYPE$JOB
  tags
}

mlflow_get_run_context.mlflow_databricks_client <- function(client, experiment_id, ...) {
  if (exists(".databricks_internals")) {
    databricks_internal_env <- get(".databricks_internals", envir = .GlobalEnv)
    notebook_info <- do.call(".get_notebook_info", list(), envir = databricks_internal_env)
    if (!is.na(notebook_info$id) && !is.na(notebook_info$path)) {
      return(list(
        client = client,
        tags = build_context_tags_from_databricks_notebook_info(notebook_info),
        experiment_id = experiment_id %||% notebook_info$id,
        ...
      ))
    }

    job_info <- if (exists(".get_job_info", envir = databricks_internal_env)) {
      do.call(".get_job_info", list(), envir = databricks_internal_env)
    } else {
      NA
    }
    if (!is.na(job_info) && !is.na(job_info$job_id)) {
      return(list(
        client = client,
        tags = build_context_tags_from_databricks_job_info(job_info),
        experiment_id = experiment_id %||% 0,
        ...
      ))
    }
    NextMethod()
  } else {
    NextMethod()
  }
}

MLFLOW_DATABRICKS_TAGS <- list(
  MLFLOW_DATABRICKS_NOTEBOOK_ID = "mlflow.databricks.notebookID",
  MLFLOW_DATABRICKS_NOTEBOOK_PATH = "mlflow.databricks.notebookPath",
  MLFLOW_DATABRICKS_WEBAPP_URL = "mlflow.databricks.webappURL",
  MLFLOW_DATABRICKS_RUN_URL = "mlflow.databricks.runURL",
  # The SHELL_JOB_ID and SHELL_JOB_RUN_ID tags are used for tracking the
  # Databricks Job ID and Databricks Job Run ID associated with an MLflow Project run
  MLFLOW_DATABRICKS_SHELL_JOB_ID = "mlflow.databricks.shellJobID",
  MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID = "mlflow.databricks.shellJobRunID",
  # The JOB_ID, JOB_RUN_ID, and JOB_TYPE tags are used for automatically recording Job
  # information when MLflow Tracking APIs are used within a Databricks Job
  MLFLOW_DATABRICKS_JOB_ID = "mlflow.databricks.jobID",
  MLFLOW_DATABRICKS_JOB_RUN_ID = "mlflow.databricks.jobRunID",
  MLFLOW_DATABRICKS_JOB_TYPE = "mlflow.databricks.jobType"
)
