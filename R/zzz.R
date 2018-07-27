.onLoad <- function(...) {
  mc <- mlflow_connect()
  mlflow_connection_active_set(mc)
}

