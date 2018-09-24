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
  } else if (!is.null(mlflow_local_server(tracking_uri)$tracking_uri)) {
    mlflow_local_server(tracking_uri)$tracking_uri
  } else {
    local_server <- mlflow_server(file_store = tracking_uri, port = mlflow_connect_port())
    mlflow_register_local_server(tracking_uri = tracking_uri, local_server = local_server)
    local_server$tracking_uri
  }

  new_mlflow_client(tracking_uri, server_url = server_url)
}

#' Create Experiment
#'
#' Creates an MLflow experiment.
#'
#' @param name The name of the experiment to create.
#' @param artifact_location Location where all artifacts for this experiment are stored. If
#'   not provided, the remote server will select an appropriate default.
#'
#' @export
mlflow_create_experiment <- function(name, artifact_location = NULL, client = NULL) {
  UseMethod("mlflow_create_experiment", client)
}

#' @export
mlflow_create_experiment.mlflow_client <- function(name, artifact_location = NULL, client = NULL) {
  name <- forge::cast_string(name)
  experiment_id <- mlflow_client_create_experiment(client, name, artifact_location)
  invisible(experiment_id)
}

#' Create Run
#'
#' reate a new run within an experiment. A run is usually a single execution of a machine learning or data ETL pipeline.
#'
#' MLflow uses runs to track Param, Metric, and RunTag, associated with a single execution.
#'
#' @param experiment_id Unique identifier for the associated experiment.
#' @param user_id User ID or LDAP for the user executing the run.
#' @param run_name Human readable name for run.
#' @param source_type Originating source for this run. One of Notebook, Job, Project, Local or Unknown.
#' @param source_name String descriptor for source. For example, name or description of the notebook, or job name.
#' @param status Current status of the run. One of RUNNING, SCHEDULE, FINISHED, FAILED, KILLED.
#' @param start_time Unix timestamp of when the run started in milliseconds.
#' @param end_time Unix timestamp of when the run ended in milliseconds.
#' @param source_version Git version of the source code used to create run.
#' @param entry_point_name Name of the entry point for the run.
#' @param tags Additional metadata for run in key-value pairs.
#' @export
mlflow_create_run <- function(
  experiment_id, user_id = NULL, run_name = NULL, source_type = NULL,
  source_name = NULL, entry_point_name = NULL, start_time = NULL, source_version = NULL,
  tags = NULL, client = NULL
) {
  UseMethod("mlflow_create_run", client)
}

#' @export
mlflow_create_run.mlflow_client <- function(
  experiment_id, user_id = NULL, run_name = NULL, source_type = NULL,
  source_name = NULL, entry_point_name = NULL, start_time = NULL, source_version = NULL,
  tags = NULL, client = NULL
) {
  tags <- if (!is.null(tags)) tags %>%
    purrr::imap(~ list(key = .y, value = .x)) %>%
    unname()

  mlflow_client_create_run(
    client, experiment_id, user_id, run_name, source_type,
    source_name, entry_point_name, start_time, source_version, tags
  )
}

#' Delete Experiment
#'
#' Mark an experiment and associated runs, params, metrics, … etc for deletion. If the
#'   experiment uses FileStore, artifacts associated with experiment are also deleted.
#'
#' @param experiment_id ID of the associated experiment This field is required.
#' @export
mlflow_delete_experiment <- function(experiment_id, client = NULL) {
  UseMethod("mlflow_delete_experiment", client)
}

#' @export
mlflow_delete_experiment.mlflow_client <- function(experiment_id, client = NULL) {
  mlflow_client_delete_experiment(client, experiment_id)
}

#' Restore Experiment
#'
#' Restore an experiment marked for deletion. This also restores associated metadata,
#'   runs, metrics, and params. If experiment uses FileStore, underlying artifacts
#'   associated with experiment are also restored.
#'
#' Throws RESOURCE_DOES_NOT_EXIST if experiment was never created or was permanently deleted.
#'
#' @param experiment_id ID of the associated experiment This field is required.
#' @export
mlflow_restore_experiment <- function(experiment_id, client = NULL) {
  UseMethod("mlflow_restore_experiment", client)
}

#' @export
mlflow_restore_experiment.mlflow_client <- function(experiment_id, client = NULL) {
  mlflow_client_restore_experiment(client, experiment_id)
}

#' Get Run
#'
#' Get meta data, params, tags, and metrics for run. Only last logged value for each metric is returned.
#'
#' @param run_uuid Unique ID for the run.
#'
#' @export
mlflow_get_run <- function(run_uuid, client = NULL) {
  UseMethod("mlflow_get_run", client)
}

#' @export
mlflow_get_run.mlflow_client <- function(run_uuid, client = NULL) {
  response <- mlflow_rest("runs", "get", client = client, query = list(run_uuid = run_uuid))
  run <- purrr::compact(response$run)
  run %>%
    purrr::map_at("info", tidy_run_info)
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
#' @return The param value as a named list.
#' @param param_name Name of the param. This field is required.
#' @export
mlflow_get_param <- function(param_name, client = NULL, ...) {
  UseMethod("mlflow_get_param", client)
}

#' @rdname mlflow_get_param
#' @export
mlflow_get_param.mlflow_client <- function(
  param_name, client = NULL, run_id, ...
) {
  response <- mlflow_client_get_param(client, run_id, param_name)
  as.data.frame(response$parameter, stringsAsFactors = FALSE)
}

#' Get Metric
#'
#' API to retrieve the logged value for a metric during a run. For a run, if this
#'   metric is logged more than once, this API will retrieve only the latest value logged.
#'
#' @param metric_key Name of the metric.
#' @export
mlflow_get_metric <- function(metric_key, client = NULL, ...) {
  UseMethod("mlflow_get_metric", client)
}

#' @rdname mlflow_get_metric
#' @param run_id Run ID.
#' @export
mlflow_get_metric.mlflow_client <- function(metric_key, client = NULL, run_id, ...) {
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
#' @export
mlflow_get_metric_history <- function(metric_key, client = NULL, ...) {
  UseMethod("mlflow_get_metric_history", client)
}

#' @rdname mlflow_get_metric_history
#' @param run_id Run ID.
#' @export
mlflow_get_metric_history.mlflow_client <- function(
  metric_key, client = NULL, run_id, ...
) {
  response <- mlflow_client_get_metric_history(client, run_id, metric_key)
  metrics <- response$metrics
  metrics$timestamp <- as.POSIXct(as.double(metrics$timestamp) / 1000, origin = "1970-01-01")
  as.data.frame(metrics, stringsAsFactors = FALSE)
}

#' List Experiments
#'
#' Get a list of all experiments.
#'
#' @param view_type Qualifier for type of experiments to be returned. Defaults to `ACTIVE_ONLY`.
#' @export
mlflow_list_experiments <- function(
  view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL"), client = NULL, ...
) {
  UseMethod("mlflow_list_experiments", client)
}

#' @rdname mlflow_list_experiments
#' @export
mlflow_list_experiments.mlflow_client <- function(
  view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL"), client = NULL, ...
) {
  view_type <- match.arg(view_type)
  mlflow_client_list_experiments(client, view_type)
}

#' Get Experiment
#'
#' Get meta data for experiment and a list of runs for this experiment.
#'
#' @param experiment_id Identifer to get an experiment.
#' @export
mlflow_get_experiment <- function(experiment_id, client = NULL, ...) {
  UseMethod("mlflow_get_experiment", client)
}

#' @rdname mlflow_get_experiment
#' @export
mlflow_get_experiment.mlflow_client <- function(experiment_id, client = NULL, ...) {
  mlflow_client_get_experiment(client, experiment_id)
}

#' Terminate a Run
#'
#' @param run_id Unique identifier for the run.
#' @param status Updated status of the run. Defaults to `FINISHED`.
#' @param end_time Unix timestamp of when the run ended in milliseconds.
#' @export
mlflow_set_terminated <- function(
  run_id, status = c("FINISHED", "SCHEDULED", "FAILED", "KILLED"),
  end_time = NULL, client
) {
  status <- match.arg(status)
  # end_time <- end_time %||% current_time()
  response <- mlflow_client_update_run(client, run_id, status, end_time)
  tidy_run_info(response$run_info)
}

#' Log Artifact
#'
#' Logs an specific file or directory as an artifact.
#'
#' @param path The file or directory to log as an artifact.
#' @param artifact_path Destination path within the run’s artifact URI.
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
  UseMethod("mlflow_log_artifact")
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
  .globals$active_experiment <- NULL
  .globals$active_run <- NULL

  invisible(uri)
}

#' Get Remote Tracking URI
#'
#' @export
mlflow_get_tracking_uri <- function() {
  .globals$tracking_uri %||% {
    env_uri <- Sys.getenv("MLFLOW_TRACKING_URI")
    if (nchar(env_uri)) env_uri else fs::path_abs("mlruns")
  }
}
