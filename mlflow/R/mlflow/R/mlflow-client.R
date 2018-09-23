new_mlflow_client <- function(tracking_uri) {
  structure(
    list(
      tracking_uri = tracking_uri
    ),
    class = "mlflow_client"
  )
}

mlflow_client <- function(tracking_uri = NULL) {
  tracking_uri <- tracking_uri %||% Sys.getenv("MLFLOW_TRACKING_URI") %||%
    stop("`tracking_uri` must be specified when `MLFLOW_TRACKING_URI` is not set.")
  new_mlflow_client(tracking_uri)
}

mlflow_client_create_experiment <- function(client, name, artifact_location) {
  response <- mlflow_rest(
    "experiments", "create", client = client, verb = "POST",
    data = list(
      name = name,
      artifact_location = artifact_location
    )
  )
  cast_scalar_integer(response$experiment_id)
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
