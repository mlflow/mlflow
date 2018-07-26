.globals <- new.env(parent = emptyenv())

mlflow_connection_active_get <- function() {
  .globals$active
}

mlflow_connection_active_set <- function(mc) {
  .globals$active <- mc
}

#' Connect to MLflow
#'
#' Connect to local or remote MLflow instance.
#'
#' @param url Optional URL to the remote MLflow server; otherwise,
#'   will launch and connect local instance.
#' @param store The root of the backing file store for
#'   experiment and run data. Defaults to \code{./mlruns}.
#' @param artifacts Local or S3 URI to store artifacts in, for
#'   newly created experiments. Note that this flag
#'   does not impact already-created experiments.
#'   Defaults to a location inside \code{store}.
mlflow_connect <- function(url = NULL,
                           store = NULL,
                           artifacts = NULL) {
  handle <- NULL

  if (is.null(url)) {
    args <- list(
      "--file-store" = store,
      "--default-artifact-root" = artifacts
    )

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
    url <- getOption("mlflow.ui", "http://127.0.0.1:5000")
  }

  mc <- structure(
    class = "mlflow_connection",
    list(
      url = url,
      handle = handle
    )
  )

  mlflow_connection_wait(mc)

  mc
}

#' Disconnect from MLflow
#'
#' Disconnects from a local MLflow instance.
#'
#' @param mc The MLflow connection created using \code{mlflow_connect()}.
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
