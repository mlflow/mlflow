
new_mlflow_client <- function(tracking_uri) {
  UseMethod("new_mlflow_client")
}

new_mlflow_uri <- function(raw_uri) {
  # Special case 'databricks'
  if (identical(raw_uri, "databricks")) {
    raw_uri <- paste0("databricks", "://")
  }

  if (!grepl("://", raw_uri)) {
    raw_uri <- paste0("file://", raw_uri)
  }
  parts <- strsplit(raw_uri, "://")[[1]]
  structure(
    list(scheme = parts[1], path = parts[2]),
    class = c(paste("mlflow_", parts[1], sep = ""), "mlflow_uri")
  )
}

new_mlflow_client_impl <- function(get_host_creds, get_cli_env = list, class = character()) {
  structure(
    list(get_host_creds = get_host_creds,
         get_cli_env = get_cli_env
    ),
    class = c(class, "mlflow_client")
  )
}

new_mlflow_host_creds <- function(host = NA, username = NA, password = NA, token = NA,
                                   insecure = "False") {
  insecure_arg <- if (is.null(insecure) || is.na(insecure)) {
    "False"
  } else {
    list(true = "True", false = "False")[[tolower(insecure)]]
  }
  structure(
    list(host = host, username = username, password = password, token = token,
         insecure = insecure_arg),
    class = "mlflow_host_creds"
  )
}

#' @export
print.mlflow_host_creds <- function(x, ...) {
  mlflow_host_creds <- x
  args <- list(
    host = if (is.na(mlflow_host_creds$host)) {
      ""
    } else {
      paste ("host = ", mlflow_host_creds$host, sep = "")
    },
    username = if (is.na(mlflow_host_creds$username)) {
      ""
    } else {
      paste ("username = ", mlflow_host_creds$username, sep = "")
    },
    password = if (is.na(mlflow_host_creds$password)) {
      ""
    } else {
      "password = *****"
    },
    token = if (is.na(mlflow_host_creds$token)) {
      ""
    } else {
      "token = *****"
    },
    insecure = paste("insecure = ", as.character(mlflow_host_creds$insecure),
                     sep = ""),
    sep = ", "
  )
  cat("mlflow_host_creds( ")
  do.call(cat, args[args != ""])
  cat(")\n")
}

new_mlflow_client.mlflow_file <- function(tracking_uri) {
  path <- tracking_uri$path
  server_url <- if (!is.null(mlflow_local_server(path)$server_url)) {
    mlflow_local_server(path)$server_url
  } else {
    local_server <- mlflow_server(file_store = path, port = mlflow_connect_port())
    mlflow_register_local_server(tracking_uri = path, local_server = local_server)
    local_server$server_url
  }
  new_mlflow_client_impl(get_host_creds = function () {
    new_mlflow_host_creds(host = server_url)
  }, class = "mlflow_file_client")
}

new_mlflow_client.default <- function(tracking_uri) {
  stop(paste("Unsupported scheme: '", tracking_uri$scheme, "'", sep = ""))
}

get_env_var <- function(x) {
  new_name <- paste("MLFLOW_TRACKING_", x, sep = "")
  res <- Sys.getenv(new_name, NA)
  if (is.na(res)) {
    old_name <- paste("MLFLOW_", x, sep = "")
    res <- Sys.getenv(old_name, NA)
    if (!is.na(res)) {
      warning(paste("'", old_name, "' is deprecated. Please use '", new_name, "' instead."),
                    sepc = "" )
    }
  }
  res
}

basic_http_client <- function(tracking_uri) {
  host <- paste(tracking_uri$scheme, tracking_uri$path, sep = "://")
  get_host_creds <- function () {
    new_mlflow_host_creds(
      host = host,
      username = get_env_var("USERNAME"),
      password = get_env_var("PASSWORD"),
      token = get_env_var("TOKEN"),
      insecure = get_env_var("INSECURE")
    )
  }
  cli_env <- function() {
    creds <- get_host_creds()
    res <- list(
      MLFLOW_TRACKING_USERNAME = creds$username,
      MLFLOW_TRACKING_PASSWORD = creds$password,
      MLFLOW_TRACKING_TOKEN = creds$token,
      MLFLOW_TRACKING_INSECURE = creds$insecure
    )
    res[!is.na(res)]
  }
  new_mlflow_client_impl(get_host_creds, cli_env, class = "mlflow_http_client")
}

new_mlflow_client.mlflow_http <- function(tracking_uri) {
  basic_http_client(tracking_uri)
}

new_mlflow_client.mlflow_https <- function(tracking_uri) {
  basic_http_client(tracking_uri)
}

#' Initialize an MLflow Client
#'
#' Initializes and returns an MLflow client that communicates with the tracking server or store
#' at the specified URI.
#'
#' @param tracking_uri The tracking URI. If not provided, defaults to the service
#'  set by `mlflow_set_tracking_uri()`.
#' @export
mlflow_client <- function(tracking_uri = NULL) {
  tracking_uri <- new_mlflow_uri(tracking_uri %||% mlflow_get_tracking_uri())
  client <- new_mlflow_client(tracking_uri)
  if (inherits(client, "mlflow_file_client")) mlflow_validate_server(client)
  client
}
