new_mlflow_client <- function(tracking_uri, server_url = NULL) {
  structure(
    list(
      tracking_uri = tracking_uri,
      server_url = server_url %||% tracking_uri
    ),
    class = "mlflow_client"
  )
}

#' Initialize an MLflow Client
#'
#' Initializes an MLflow client.
#'
#' @param tracking_uri The tracking URI. If not provided, defaults to the service
#'  set by `mlflow_set_tracking_uri()`.
#' @keywords internal
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

#' Create Experiment - Tracking Client
#'
#' Creates an MLflow experiment.
#'
#' @param name The name of the experiment to create.
#' @param artifact_location Location where all artifacts for this experiment are stored. If
#'   not provided, the remote server will select an appropriate default.
#' @template roxlate-client
mlflow_client_create_experiment <- function(client, name, artifact_location = NULL) {
  name <- forge::cast_string(name)
  response <- mlflow_rest(
    "experiments", "create", client = client, verb = "POST",
    data = list(
      name = name,
      artifact_location = artifact_location
    )
  )
  invisible(response$experiment_id)
}

#' List Experiments
#'
#' Gets a list of all experiments.
#'
#' @param view_type Qualifier for type of experiments to be returned. Defaults to `ACTIVE_ONLY`.
#' @template roxlate-client
mlflow_client_list_experiments <- function(client, view_type = c("ACTIVE_ONLY", "DELETED_ONLY", "ALL")) {
  view_type <- match.arg(view_type)
  response <- mlflow_rest(
    "experiments", "list", client = client, verb = "GET",
    query = list(
      view_type = view_type
    ))
  exps <- response$experiments

  exps$artifact_location <- mlflow_relative_paths(exps$artifact_location)
  exps
}

#' Get Experiment
#'
#' Gets metadata for an experiment and a list of runs for the experiment.
#'
#' @param experiment_id Identifer to get an experiment.
#' @template roxlate-client
mlflow_client_get_experiment <- function(client, experiment_id) {
  mlflow_rest(
    "experiments", "get", client = client,
    query = list(experiment_id = experiment_id)
  )
}

#' Get Experiment by Name
#'
#' Get meta data for experiment by name.
#'
#' @param name The experiment name.
#' @template roxlate-client
mlflow_client_get_experiment_by_name <- function(client, name) {
  exps <- mlflow_client_list_experiments(client = client)
  experiment <- exps[exps$name == name, ]
  if (nrow(experiment)) experiment else NULL
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
#' @param start_time Unix timestamp of when the run started in milliseconds.
#' @param source_version Git version of the source code used to create run.
#' @param entry_point_name Name of the entry point for the run.
#' @param tags Additional metadata for run in key-value pairs.
#' @template roxlate-client
mlflow_client_create_run <- function(
  client, experiment_id, user_id = NULL, run_name = NULL, source_type = NULL,
  source_name = NULL, entry_point_name = NULL, start_time = NULL,
  source_version = NULL, tags = NULL
) {
  tags <- if (!is.null(tags)) tags %>%
    purrr::imap(~ list(key = .y, value = .x)) %>%
    unname()

  start_time <- start_time %||% current_time()
  user_id <- user_id %||% mlflow_user()

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
  new_mlflow_entities_run(response)
}

mlflow_rest_update_run <- function(client, run_uuid, status, end_time) {
  mlflow_rest("runs", "update", client = client, verb = "POST", data = list(
    run_uuid = run_uuid,
    status = status,
    end_time = end_time
  ))
}

#' Delete Experiment
#'
#' Marks an experiment and associated runs, params, metrics, etc. for deletion. If the
#'   experiment uses FileStore, artifacts associated with experiment are also deleted.
#'
#' @param experiment_id ID of the associated experiment. This field is required.
#' @template roxlate-client
mlflow_client_delete_experiment <- function(client, experiment_id) {
  mlflow_rest(
    "experiments", "delete", client = client, verb = "POST",
    data = list(experiment_id = experiment_id),
  )
}

#' Restore Experiment
#'
#' Restores an experiment marked for deletion. This also restores associated metadata,
#'   runs, metrics, and params. If experiment uses FileStore, underlying artifacts
#'   associated with experiment are also restored.
#'
#' Throws `RESOURCE_DOES_NOT_EXIST` if the experiment was never created or was permanently deleted.
#'
#' @param experiment_id ID of the associated experiment. This field is required.
#' @template roxlate-client
mlflow_client_restore_experiment <- function(client, experiment_id) {
  mlflow_rest(
    "experiments", "restore", client = client, verb = "POST",
    data = list(experiment_id = experiment_id),
  )
}

#' Get Run
#'
#' Gets metadata, params, tags, and metrics for a run. Only last logged value for each metric is returned.
#'
#' @template roxlate-run-id
#' @template roxlate-client
mlflow_client_get_run <- function(client, run_id) {
  response <- mlflow_rest(
    "runs", "get", client = client, verb = "GET",
    query = list(run_uuid = run_id)
  )
  new_mlflow_entities_run(response)
}

#' Log Metric
#'
#' Logs a metric for a run. Metrics key-value pair that records a single float measure.
#'   During a single execution of a run, a particular metric can be logged several times.
#'   Backend will keep track of historical values along with timestamps.
#'
#' @param key Name of the metric.
#' @param value Float value for the metric being logged.
#' @param timestamp Unix timestamp in milliseconds at the time metric was logged.
#' @template roxlate-run-id
#' @template roxlate-client
mlflow_client_log_metric <- function(client, run_id, key, value, timestamp = NULL) {
  if (!is.numeric(value)) stop(
    "Metric `", key, "`` must be numeric but ", class(value)[[1]], " found.",
    call. = FALSE
  )
  timestamp <- timestamp %||% current_time()
  mlflow_rest("runs", "log-metric", client = client, verb = "POST", data = list(
    run_uuid = run_id,
    key = key,
    value = value,
    timestamp = timestamp
  ))
}

#' Log Parameter
#'
#' Logs a parameter for a run. Examples are params and hyperparams
#'   used for ML training, or constant dates and values used in an ETL pipeline.
#'   A param is a STRING key-value pair. For a run, a single parameter is allowed
#'   to be logged only once.
#'
#' @param key Name of the parameter.
#' @param value String value of the parameter.
#' @template roxlate-run-id
#' @template roxlate-client
mlflow_client_log_param <- function(client, run_id, key, value) {
  mlflow_rest("runs", "log-parameter", client = client, verb = "POST", data = list(
    run_uuid = run_id,
    key = key,
    value = cast_string(value)
  ))
}

#' Set Tag
#'
#' Sets a tag on a run. Tags are run metadata that can be updated during a run and
#'  after a run completes.
#'
#' @param key Name of the tag. Maximum size is 255 bytes. This field is required.
#' @param value String value of the tag being logged. Maximum size is 500 bytes. This field is required.
#' @template roxlate-run-id
#' @template roxlate-client
mlflow_client_set_tag <- function(client, run_id, key, value) {
  mlflow_rest("runs", "set-tag", client = client, verb = "POST", data = list(
    run_uuid = run_id,
    key = key,
    value = value
  ))
  invisible(NULL)
}

#' Terminate a Run
#'
#' Terminates a run.
#'
#' @param run_id Unique identifier for the run.
#' @param status Updated status of the run. Defaults to `FINISHED`.
#' @param end_time Unix timestamp of when the run ended in milliseconds.
#' @template roxlate-run-id
#' @template roxlate-client
mlflow_client_set_terminated <- function(
  client, run_id, status = c("FINISHED", "SCHEDULED", "FAILED", "KILLED"),
  end_time = NULL
) {
  status <- match.arg(status)
  end_time <- end_time %||% current_time()
  response <- mlflow_rest_update_run(client, run_id, status, end_time)
  tidy_run_info(response$run_info)
}

#' Delete a Run
#'
#' @template roxlate-client
#' @template roxlate-run-id
mlflow_client_delete_run <- function(client, run_id) {
  mlflow_rest("runs", "delete", client = client, verb = "POST", data = list(
    run_uuid = run_id
  ))
}

#' Restore a Run
#'
#' @template roxlate-client
#' @template roxlate-run-id
mlflow_client_restore_run <- function(client, run_id) {
  mlflow_rest("runs", "restore", client = client, verb = "POST", data = list(
    run_uuid = run_id
  ))
}

#' Log Artifact
#'
#' Logs a specific file or directory as an artifact for a run.
#'
#' @param path The file or directory to log as an artifact.
#' @param artifact_path Destination path within the run's artifact URI.
#' @template roxlate-client
#' @template roxlate-run-id
#'
#' @details
#'
#' When logging to Amazon S3, ensure that the user has a proper policy
#' attached to it, for instance:
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
mlflow_client_log_artifact <- function(client, run_id, path, artifact_path = NULL) {
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

#' List Artifacts
#'
#' Gets a list of artifacts.
#'
#' @template roxlate-client
#' @template roxlate-run-id
#' @param path The run's relative artifact path to list from. If not specified, it is
#'  set to the root artifact path
mlflow_client_list_artifacts <- function(client, run_id, path = NULL) {
  response <- mlflow_rest(
    "artifacts", "list", client = client, verb = "GET",
    query = list(
      run_uuid = run_id,
      path = path
    ))
  response
}

#' Download Artifacts
#'
#' Download an artifact file or directory from a run to a local directory if applicable,
#'   and return a local path for it.
#'
#' @template roxlate-client
#' @template roxlate-run-id
#' @param path Relative source path to the desired artifact.
mlflow_client_download_artifacts <- function(client, run_id, path) {
  result <- mlflow_cli(
    "artifacts", "download",
    "--run-id", run_id,
    "--artifact-path", path,
    echo = FALSE,
    stderr_callback = function(x, p) {
      if (grepl("FileNotFoundError", x))
        stop(
          gsub("(.|\n)*(?=FileNotFoundError)", "", x, perl = TRUE),
          call. = FALSE
        )
    }
  )

  gsub("\n", "", result$stdout)
}
