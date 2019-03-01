# Utils for databricks authentication

new_mlflow_client.databricks <- function(scheme, tracking_uri) {
  parts <- strsplit(tracking_uri, "://")
  if (scheme != "databricks" || parts[[1]][1] != "databricks") {
    stop(paste("Unexpected tracking uri '", tracking_uri, "' for scheme '", scheme, "'", sep = ""))
  }
  profile <- parts[[1]][2]
  # make sure we can read the config
  config <- get_databricks_config(profile)
  new_mlflow_client_impl(
    get_host_creds = function() {
      get_databricks_config(profile)
    },
    cli_env = function() {
      databricks_config_as_env(get_databricks_config(profile))
    },
    clazz = class(scheme)
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
#' @importFrom utils hasName
get_databricks_config_for_profile <- function(profile) {
  config_path <- Sys.getenv("DATABRICKS_CONFIG_FILE", NA)
  config_path <- if (is.na(config_path)) path.expand("~/.databrickscfg") else config_path
  if (!file.exists(config_path)){
    stop(paste("Databricks configuration file is missing. Expected config file ", config_path))
  }
  config <- read.ini(config_path)
  if (!hasName(config, profile)) {
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
  } else if (exists("databricks_authentication_provider")) {
    do.call("databricks_authentication_provider", list())
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
