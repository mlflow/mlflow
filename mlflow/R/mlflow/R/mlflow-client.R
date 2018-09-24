new_mlflow_client <- function(tracking_uri, server_url = NULL) {
  structure(
    list(
      tracking_uri = tracking_uri,
      server_url = server_url %||% tracking_uri
    ),
    class = "mlflow_client"
  )
}

#' Initialize an MLflow client
#'
#' @param tracking_uri The tracking URI
#'
#' @export
mlflow_client <- function(tracking_uri = NULL) {
  tracking_uri <- tracking_uri %||% mlflow_tracking_uri()
  if (!startsWith(tracking_uri, "http") && is.null(mlflow_local_server(tracking_uri))) {
    local_server <- mlflow_server(file_store = tracking_uri, port = mlflow_connect_port())
    mlflow_register_local_server(tracking_uri = tracking_uri, local_server = local_server)
  }
  new_mlflow_client(tracking_uri, server_url = mlflow_local_server(tracking_uri)$tracking_uri)
}

mlflow_client_create_experiment <- function(client, name, artifact_location) {
  response <- mlflow_rest(
    "experiments", "create", client = client, verb = "POST",
    data = list(
      name = name,
      artifact_location = artifact_location
    )
  )
  response$experiment_id
}

mlflow_client_create_run <- function(
  client, experiment_id, user_id, run_name, source_type,
  source_name, entry_point_name, start_time, source_version, tags
) {
  response <- mlflow_rest(
    "runs", "create", client = client, verb = "POST",
    data = list(
      experiment_id = experiment_id,
      user_id = user_id,
      run_name = run_name,
      source_type = source_type,
      source_name = source_name,
      entry_point_name = entry_point_name,
      start_time = start_time,
      source_version = source_version,
      tags = tags
    )
  )
  response
}

mlflow_client_delete_experiment <- function(client, experimend_id) {
  response <- mlflow_rest(
    "experiments", "delete", client = client, verb = "POST",
    data = list(experiment_id = experiment_id),
  )
  response
}

mlflow_client_restore_experiment <- function(client, experiment_id) {
  response <- mlflow_rest(
    "experiments", "restore", client = client, verb = "POST",
    data = list(experiment_id = experiment_id),
  )
  response
}

mlflow_client_get_run <- function(client, run_uuid) {
  response <- mlflow_rest(
    "runs", "get", client = client, verb = "GET",
    data = list(run_uuid = run_uuid),
  )
  response
}

mlflow_client_log_metric <- function(client, run_uuid, key, value, timestamp) {
  response <- mlflow_rest("runs", "log-metric", verb = "POST", data = list(
    run_uuid = run_uuid,
    key = key,
    value = value,
    timestamp = timestamp
  ))
  response
}

mlflow_client_set_tag <- function(client, run_uuid, key, value) {
  response <- mlflow_rest("runs", "set-tag", verb = "POST", data = list(
    run_uuid = run_uuid,
    key = key,
    value = value
  ))
  response
}

mlflow_client_log_param <- function(client, run_uuid, key, value) {
  response <- mlflow_rest("runs", "log-parameter", verb = "POST", data = list(
    run_uuid = run_uuid,
    key = key,
    value = cast_string(value)
  ))
  response
}

mlflow_client_get_param <- function(client, run_uuid, param_name) {
  response <- mlflow_rest("params", "get", query = list(
    run_uuid = run_uuid,
    param_name = param_name
  ))
  response
}
