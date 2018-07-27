.onLoad <- function(...) {
  if (getOption("mlflow.autoconnect", TRUE) &&
      !mlflow_tracking_is_remote() &&
      mlflow_is_installed())
  {
    mc <- mlflow_connect()
    mlflow_connection_active_set(mc)
  }
}

