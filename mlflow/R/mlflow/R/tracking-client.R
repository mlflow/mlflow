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
  tracking_uri <- tracking_uri %||% mlflow_get_tracking_uri()
  server_url <- if (startsWith(tracking_uri, "http")) {
    tracking_uri
  } else if (!is.null(mlflow_local_server(tracking_uri)$server_url)) {
    mlflow_local_server(tracking_uri)$server_url
  } else {
    local_server <- mlflow_server(file_store = tracking_uri, port = mlflow_connect_port())
    mlflow_register_local_server(tracking_uri = tracking_uri, local_server = local_server)
    local_server$server_url
  }

  new_mlflow_client(tracking_uri, server_url = server_url)
}

#' Log Metric
#'
#' API to log a metric for a run. Metrics key-value pair that record a single float measure.
#'   During a single execution of a run, a particular metric can be logged several times.
#'   Backend will keep track of historical values along with timestamps.
#'
#' @param key Name of the metric.
#' @param value Float value for the metric being logged.
#' @param timestamp Unix timestamp in milliseconds at the time metric was logged.
#' @template roxlate-client-optional
#' @export
mlflow_log_metric <- function(key, value, timestamp = NULL, client = NULL, ...) {
  UseMethod("mlflow_log_metric", client)
}

#' @rdname mlflow_log_metric
#' @param run_id Run ID.
#' @export
mlflow_log_metric.mlflow_client <- function(
  key, value, timestamp = NULL, client = NULL, run_id, ...
) {
  if (!rlang::inherits_any(value, c("character", "numeric", "integer"))) {
    stop("Metric ", key, " must be a character or numeric but ", class(value), " found.")
  }
  timestamp <- timestamp %||% current_time()
  mlflow_client_log_metric(
    client, run_uuid = run_id, key = key, value = value, timestamp = timestamp
  )
}

#' Set Tag
#'
#' Set a tag on a run. Tags are run metadata that can be updated during and
#'  after a run completes.
#'
#' @param key Name of the tag. Maximum size is 255 bytes. This field is required.
#' @param value String value of the tag being logged. Maximum size is 500 bytes. This field is required.
#' @template roxlate-client-optional
#' @export
mlflow_set_tag <- function(key, value, client = NULL, ...) {
  UseMethod("mlflow_set_tag", client)
}

#' @rdname mlflow_set_tag
#' @param run_id Run ID.
#' @export
mlflow_set_tag.mlflow_client <- function(key, value, client = NULL, run_id, ...) {
  mlflow_client_set_tag(client, run_uuid = run_id, key = key, value = value)
}

#' Log Parameter
#'
#' API to log a parameter used for this run. Examples are params and hyperparams
#'   used for ML training, or constant dates and values used in an ETL pipeline.
#'   A params is a STRING key-value pair. For a run, a single parameter is allowed
#'   to be logged only once.
#'
#' @param key Name of the parameter.
#' @param value String value of the parameter.
#' @template roxlate-client-optional
#' @export
mlflow_log_param <- function(key, value, client = NULL, ...) {
  UseMethod("mlflow_log_param", client)
}

#' @rdname mlflow_log_param
#' @param run_id Run ID.
#' @export
mlflow_log_param.mlflow_client <- function(
  key, value, client = NULL, run_id, ...
) {
  mlflow_client_log_param(client, run_uuid = run_id, key, value)
}

#' Get Param
#'
#' Get a param value.
#'
#' @return The param name and value as a data frame.
#' @param param_name Name of the param. This field is required.
#' @param run_id Run ID.
#' @template roxlate-client
#' @export
mlflow_get_param <- function(param_name, client, run_id) {
  response <- mlflow_client_get_param(client, run_id, param_name)
  as.data.frame(response$parameter, stringsAsFactors = FALSE)
}

#' Get Metric
#'
#' API to retrieve the logged value for a metric during a run. For a run, if this
#'   metric is logged more than once, this API will retrieve only the latest value logged.
#'
#' @param metric_key Name of the metric.
#' @param run_id Run ID.
#' @template roxlate-client
#' @export
mlflow_get_metric <- function(metric_key, client, run_id) {
  response <- mlflow_client_get_metric(client, run_id, metric_key)
  metric <- response$metric
  metric$timestamp <- as.POSIXct(as.double(metric$timestamp) / 1000, origin = "1970-01-01")
  as.data.frame(metric, stringsAsFactors = FALSE)
}

#' Get Metric History
#'
#' For cases that a metric is logged more than once during a run, this API can be used
#'   to retrieve all logged values for this metric.
#'
#' @param metric_key Name of the metric.
#' @param run_id Run ID.
#' @template roxlate-client
#' @export
mlflow_get_metric_history <- function(metric_key, client, run_id) {
  response <- mlflow_client_get_metric_history(client, run_id, metric_key)
  metrics <- response$metrics
  metrics$timestamp <- as.POSIXct(as.double(metrics$timestamp) / 1000, origin = "1970-01-01")
  as.data.frame(metrics, stringsAsFactors = FALSE)
}

#' Terminate a Run
#'
#' @param run_id Unique identifier for the run.
#' @param status Updated status of the run. Defaults to `FINISHED`.
#' @param end_time Unix timestamp of when the run ended in milliseconds.
#' @template roxlate-client
#' @export
mlflow_set_terminated <- function(
  run_id, status = c("FINISHED", "SCHEDULED", "FAILED", "KILLED"),
  end_time = NULL, client
) {
  status <- match.arg(status)
  response <- mlflow_client_update_run(client, run_id, status, end_time)
  tidy_run_info(response$run_info)
}

#' Log Artifact
#'
#' Logs an specific file or directory as an artifact.
#'
#' @param path The file or directory to log as an artifact.
#' @param artifact_path Destination path within the run’s artifact URI.
#' @template roxlate-client-optional
#'
#' @details
#'
#' When logging to Amazon S3, ensure that the user has a proper policy
#' attach to it, for instance:
#'
#' \code{
#' {
#' "Version": "2012-10-17",
#' "Statement": [
#'   {
#'     "Sid": "VisualEditor0",
#'     "Effect": "Allow",
#'     "Action": [
#'       "s3:PutObject",
#'       "s3:GetObject",
#'       "s3:ListBucket",
#'       "s3:GetBucketLocation"
#'       ],
#'     "Resource": [
#'       "arn:aws:s3:::mlflow-test/*",
#'       "arn:aws:s3:::mlflow-test"
#'       ]
#'   }
#'   ]
#' }
#' }
#'
#' Additionally, at least the \code{AWS_ACCESS_KEY_ID} and \code{AWS_SECRET_ACCESS_KEY}
#' environment variables must be set to the corresponding key and secrets provided
#' by Amazon IAM.
#'
#' @export
mlflow_log_artifact <- function(path, artifact_path = NULL, client = NULL, ...) {
  UseMethod("mlflow_log_artifact", client)
}

#' @rdname mlflow_log_artifact
#' @param run_id The run associated with this artifact.
#' @export
mlflow_log_artifact.mlflow_client <- function(path, artifact_path = NULL, client = NULL, run_id = NULL, ...) {
  artifact_param <- NULL
  if (!is.null(artifact_path)) artifact_param <- "--artifact-path"

  if (as.logical(fs::is_file(path))) {
    command <- "log-artifact"
    local_param <- "--local-file"
  } else {
    command <- "log-artifacts"
    local_param <- "--local-dir"
  }

  mlflow_cli("artifacts",
             command,
             local_param,
             path,
             artifact_param,
             artifact_path,
             "--run-id",
             run_id)

  invisible(NULL)
}
