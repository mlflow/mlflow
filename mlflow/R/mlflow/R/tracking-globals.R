#' @include tracking-observer.R
NULL

mlflow_push_active_run_id <- function(run_id) {
  .globals$active_run_stack <- c(.globals$active_run_stack, run_id)
  mlflow_register_tracking_event("active_run_id", list(run_id = run_id))
}

mlflow_pop_active_run_id <- function() {
  .globals$active_run_stack <- .globals$active_run_stack[1:length(.globals$active_run_stack) - 1]
}

mlflow_get_active_run_id <- function() {
  if (length(.globals$active_run_stack) == 0) {
    NULL
  } else {
    .globals$active_run_stack[length(.globals$active_run_stack)]
  }
}

mlflow_set_active_experiment_id <- function(experiment_id) {
  .globals$active_experiment_id <- experiment_id
  mlflow_register_tracking_event(
    "active_experiment_id", list(experiment_id = experiment_id)
  )
}

mlflow_get_active_experiment_id <- function() {
  .globals$active_experiment_id
}

#' Set Remote Tracking URI
#'
#' Specifies the URI to the remote MLflow server that will be used
#' to track experiments.
#'
#' @param uri The URI to the remote MLflow server.
#'
#' @export
mlflow_set_tracking_uri <- function(uri) {
  .globals$tracking_uri <- uri
  mlflow_register_tracking_event("tracking_uri", list(uri = uri))

  invisible(uri)
}

#' Get Remote Tracking URI
#'
#' Gets the remote tracking URI.
#'
#' @export
mlflow_get_tracking_uri <- function() {
  .globals$tracking_uri %||% {
    env_uri <- Sys.getenv("MLFLOW_TRACKING_URI")
    if (nchar(env_uri)) env_uri else paste("file://", fs::path_abs("mlruns"), sep = "")
  }
}

#' Set Model Registry URI
#'
#' Specifies the URI to the model registry service used for model registry operations.
#'
#' @param uri The URI to the model registry service.
#' @details For Databricks model registry, use `databricks-uc` or
#'   `databricks-uc://<profile>`. When the tracking URI is `databricks`,
#'   this is the default registry URI.
#'
#' @export
mlflow_set_registry_uri <- function(uri) {
  .globals$registry_uri <- uri
  mlflow_register_tracking_event("registry_uri", list(uri = uri))

  invisible(uri)
}

#' Get Model Registry URI
#'
#' Gets the model registry URI.
#'
#' @export
mlflow_get_registry_uri <- function() {
  mlflow_resolve_registry_uri(tracking_uri = mlflow_get_tracking_uri())
}

mlflow_uri_is_set <- function(uri) {
  is.character(uri) && length(uri) == 1L && !is.na(uri) && nzchar(uri)
}

mlflow_default_registry_uri <- function(tracking_uri) {
  if (identical(tracking_uri, "databricks")) {
    return("databricks-uc")
  }
  if (identical(tracking_uri, "databricks-uc")) {
    return("databricks-uc")
  }
  if (startsWith(tracking_uri, "databricks://")) {
    return(sub("^databricks://", "databricks-uc://", tracking_uri))
  }
  if (startsWith(tracking_uri, "databricks-uc://")) {
    return(tracking_uri)
  }
  tracking_uri
}

mlflow_resolve_registry_uri <- function(tracking_uri, registry_uri = NULL) {
  if (mlflow_uri_is_set(registry_uri)) {
    return(registry_uri)
  }
  if (!is.null(.globals$registry_uri)) {
    if (mlflow_uri_is_set(.globals$registry_uri)) {
      return(.globals$registry_uri)
    }
    return(mlflow_default_registry_uri(tracking_uri))
  }
  env_uri <- Sys.getenv("MLFLOW_REGISTRY_URI")
  if (nchar(env_uri)) {
    return(env_uri)
  }
  mlflow_default_registry_uri(tracking_uri)
}
