mlflow_rest_path <- function(version) {
  switch(
    version,
    "2.0" = "ajax-api/2.0/preview/mlflow"
  )
}

mlflow_rest_body <- function(data) {
  data <- Filter(length, data)
  paste0(
    "\"",
    gsub(
      "\\\"",
      "\\\\\"",
      as.character(
        jsonlite::toJSON(data, auto_unbox = TRUE)
      )
    ),
    "\""
  )
}

#' @importFrom httr add_headers
mlflow_rest_headers <- function() {
  add_headers("Content-Type" = "application/json")
}

#' @importFrom httr timeout
mlflow_rest_timeout <- function() {
  timeout(getOption("mlflow.rest.timeout", 1))
}

#' @importFrom httr content
#' @importFrom httr GET
#' @importFrom httr POST
#' @importFrom jsonlite fromJSON
#' @importFrom xml2 as_list
mlflow_rest <- function(..., client, query = NULL, data = NULL, verb = "GET", version = "2.0") {
  args <- list(...)
  tracking_url <- client$server_url

  api_url <- file.path(
    tracking_url,
    mlflow_rest_path(version),
    paste(args, collapse = "/")
  )

  response <- switch(
    verb,
    GET = GET(api_url, query = query, mlflow_rest_timeout()),
    POST = POST(api_url,
                body = mlflow_rest_body(data),
                mlflow_rest_headers(),
                mlflow_rest_timeout()),
    stop("Verb '", verb, "' is unsupported.")
  )

  if (identical(response$status_code, 500L)) {
    stop(as_list(content(response))$html$body$p[[1]])
  }

  text <- content(response, "text", encoding = "UTF-8")
  jsonlite::fromJSON(text)
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

mlflow_client_list_experiments <- function(client, view_type) {
  response <- mlflow_rest(
    "experiments", "list", client = client, verb = "GET",
    query = list(
      view_type = view_type
    ))
  exps <- response$experiments

  exps$artifact_location <- mlflow_relative_paths(exps$artifact_location)
  exps
}

mlflow_client_get_experiment <- function(client, experiment_id) {
  response <- mlflow_rest(
    "experiments", "get", client = client,
    query = list(experiment_id = experiment_id)
  )
  response
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

mlflow_client_update_run <- function(client, run_uuid, status, end_time) {
  response <- mlflow_rest("runs", "update", client = client, verb = "POST", data = list(
    run_uuid = run_uuid,
    status = status,
    end_time = end_time
  ))

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
    query = list(run_uuid = run_uuid),
  )
  response
}

mlflow_client_log_metric <- function(client, run_uuid, key, value, timestamp) {
  response <- mlflow_rest("runs", "log-metric", client = client, verb = "POST", data = list(
    run_uuid = run_uuid,
    key = key,
    value = value,
    timestamp = timestamp
  ))
  response
}

mlflow_client_set_tag <- function(client, run_uuid, key, value) {
  response <- mlflow_rest("runs", "set-tag", client = client, verb = "POST", data = list(
    run_uuid = run_uuid,
    key = key,
    value = value
  ))
  response
}

mlflow_client_log_param <- function(client, run_uuid, key, value) {
  response <- mlflow_rest("runs", "log-parameter", client = client, verb = "POST", data = list(
    run_uuid = run_uuid,
    key = key,
    value = cast_string(value)
  ))
  response
}

mlflow_client_get_param <- function(client, run_uuid, param_name) {
  response <- mlflow_rest("params", "get", client = client, query = list(
    run_uuid = run_uuid,
    param_name = param_name
  ))
  response
}

mlflow_client_get_metric <- function(client, run_uuid, metric_key) {
  response <- mlflow_rest("metrics", "get", client = client, query = list(
    run_uuid = run_uuid,
    metric_key = metric_key
  ))
  response
}

mlflow_client_get_metric_history <- function(client, run_uuid, metric_key) {
  response <- mlflow_rest("metrics", "get-history", client = client, query = list(
    run_uuid = run_uuid,
    metric_key = metric_key
  ))

  response
}
