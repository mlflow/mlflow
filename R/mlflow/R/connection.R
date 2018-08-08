mlflow_get_or_create_active_connection <- function() {
  if (is.null(mlflow_active_connection())) {
    mc <- mlflow_connect()
    mlflow_set_active_connection(mc)
  }

  mlflow_active_connection()
}

mlflow_active_connection <- function() {
  .globals$active_connection
}

mlflow_set_active_connection <- function(mc) {
  .globals$active_connection <- mc
  invisible(NULL)
}

#' @importFrom openssl rand_num
mlflow_connect_port <- function() {
  getOption(
    "mlflow.port",
    floor(5000 + rand_num(1) * 1000)
  )
}

mlflow_cli_param <- function(args, param, value)
{
  if (!is.null(value)) {
    args <- c(
      args,
      param,
      value
    )
  }

  args
}

#' Run the MLflow Tracking Server
#'
#' Wrapper for `mlflow server`.
#'
#' @param file_store The root of the backing file store for experiment and run data.
#' @param default_artifact_root Local or S3 URI to store artifacts in, for newly created experiments.
#' @param host The network address to listen on (default: 127.0.0.1).
#' @param port The port to listen on (default: 5000).
#' @param workers Number of gunicorn worker processes to handle requests (default: 4).
#' @param static_prefix A prefix which will be prepended to the path of all static paths.
#' @export
mlflow_server <- function(file_store = "mlruns", default_artifact_root = NULL,
                          host = "127.0.0.1", port = 5000, workers = 4, static_prefix = NULL) {

  file_store <- fs::path_abs(file_store)

  args <- mlflow_cli_param(list(), "--port", port) %>%
    mlflow_cli_param("--file-store", file_store) %>%
    mlflow_cli_param("--default-artifact-root", default_artifact_root) %>%
    mlflow_cli_param("--host", host) %>%
    mlflow_cli_param("--port", port) %>%
    mlflow_cli_param("--workers", workers) %>%
    mlflow_cli_param("--static-prefix", static_prefix)

  handle <- do.call(
    "mlflow_cli",
    c(
      "server",
      args,
      list(
        background = getOption("mlflow.ui.background", TRUE)
      )
    )
  )

  url <- getOption("mlflow.ui", paste(host, port, sep = ":"))
  new_mlflow_connection(url, handle, file_store = file_store)
}

#' Connect to MLflow
#'
#' Connect to local or remote MLflow instance.
#'
#' @param x (Optional) Either a URL to the remote MLflow server or the file store,
#'   i.e. the root of the backing file store for experiment and run data. If not
#'   specified, will launch and connect to a local instance listening on a random port.
#' @param activate Whether to set the connction as the active connection, defaults to `TRUE`.
#' @param ... Optional arguments passed to `mlflow_server()`.
#' @export
mlflow_connect <- function(x = NULL, activate = TRUE, ...) {
  mc <- if (!is.null(x) && startsWith(x, "http")) {
    new_mlflow_connection(x = x, handle = handle)
  } else {
    dots <- list(...)
    dots[["port"]] <- dots[["port"]] %||% mlflow_connect_port()
    if (!is.null(dots[["file_store"]]) && !is.null(x))
      stop("`x` and `file_store` cannot both be specified.", call. = FALSE)
    dots[["file_store"]] <- dots[["file_store"]] %||% x
    do.call(mlflow_server, dots)
  }

  if (activate) mlflow_set_active_connection(mc)

  mc
}

new_mlflow_connection <- function(url, handle, ...) {
  mc <- structure(
    list(
      url = if (startsWith(url, "http")) url else paste0("http://", url),
      handle = handle,
      ...
    ),
    class = "mlflow_connection"
  )

  mlflow_connection_wait(mc)
  mc
}

#' Disconnect from MLflow
#'
#' Disconnects from a local MLflow instance.
#'
#' @param mc The MLflow connection created using \code{mlflow_connect()}.
#' @export
mlflow_disconnect <- function(mc) {
  if (mc$handle$is_alive()) invisible(mc$handle$kill())
}

mlflow_connection_url <- function(mc) {
  mc$url
}

mlflow_connection_wait <- function(mc) {
  wait_for(
    function() mlflow_api(mc, "experiments", "list"),
    getOption("mlflow.connect.wait", 5),
    getOption("mlflow.connect.sleep", 1)
  )
}
