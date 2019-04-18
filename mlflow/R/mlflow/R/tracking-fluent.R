#' Create Experiment
#'
#' Creates an MLflow experiment.
#'
#' @param name The name of the experiment to create.
#' @param artifact_location Location where all artifacts for this experiment are stored. If
#'   not provided, the remote server will select an appropriate default.
#' @template roxlate-fluent
#'
#' @export
mlflow_create_experiment <- function(name, artifact_location = NULL) {
  client <- mlflow_client()
  mlflow_client_create_experiment(client, name, artifact_location)
}

#' Set Experiment
#'
#' Sets an experiment as the active experiment. If the experiment does not exist,
#'   creates an experiment with provided name.
#'
#' @param experiment_name Name of experiment to be activated.
#' @template roxlate-fluent
#' @export
mlflow_set_experiment <- function(experiment_name) {
  client <- mlflow_client()
  experiment <- mlflow_client_get_experiment_by_name(client, experiment_name)
  exp_id <- if (!is.null(experiment)) {
    experiment$experiment_id
  } else {
    message("`", experiment_name, "` does not exist. Creating a new experiment.")
    mlflow_client_create_experiment(client, experiment_name)
  }
  mlflow_set_active_experiment_id(exp_id)
}

#' Start Run
#'
#' Starts a new run within an experiment, should be used within a \code{with} block.
#'
#' @param run_uuid If specified, get the run with the specified UUID and log metrics
#'   and params under that run. The run's end time is unset and its status is set to
#'   running, but the run's other attributes remain unchanged.
#' @param experiment_id Used only when `run_uuid` is unspecified. ID of the experiment under
#'   which to create the current run. If unspecified, the run is created under
#'   a new experiment with a randomly generated name.
#' @param source_name Name of the source file or URI of the project to be associated with the run.
#'   Defaults to the current file if none provided.
#' @param source_version Optional Git commit hash to associate with the run.
#' @param entry_point_name Optional name of the entry point for to the current run.
#' @param source_type Integer enum value describing the type of the run  ("local", "project", etc.).
#' @template roxlate-fluent
#'
#' @examples
#' \dontrun{
#' with(mlflow_start_run(), {
#'   mlflow_log("test", 10)
#' })
#' }
#'
#' @export
mlflow_start_run <- function(run_uuid = NULL, experiment_id = NULL, source_name = NULL,
                             source_version = NULL, entry_point_name = NULL,
                             source_type = "LOCAL") {
  active_run <- mlflow_active_run()
  if (!is.null(active_run)) {
    stop("Run with UUID ", active_run_id(), " is already active.",
         call. = FALSE)
  }

  existing_run_uuid <- run_uuid %||% {
    env_run_id <- Sys.getenv("MLFLOW_RUN_ID")
    if (nchar(env_run_id)) env_run_id
  }

  run <- if (!is.null(existing_run_uuid)) {
    client <- mlflow_client()
    mlflow_client_get_run(client, existing_run_uuid)
  } else {
    experiment_id <- mlflow_infer_experiment_id(experiment_id)
    client <- mlflow_client()
    args <- mlflow_get_run_context(
      client,
      experiment_id = experiment_id,
      source_name = source_name,
      source_version = source_version,
      entry_point_name = entry_point_name,
      source_type = source_type
    )
    do.call(mlflow_client_create_run, args)
  }
  mlflow_set_active_run(run)
}


mlflow_get_run_context <- function(client, ...) {
  UseMethod("mlflow_get_run_context")
}

mlflow_get_run_context.default <- function(client, source_name, source_version, experiment_id,
                                           ...) {
  list(client = client,
       source_name = source_name %||% get_source_name(),
       source_version = source_version %||% get_source_version(),
       experiment_id = experiment_id %||% 0,
       ...)
}


#' Log Metric
#'
#' Logs a metric for this run. Metrics key-value pair that records a single float measure.
#'   During a single execution of a run, a particular metric can be logged several times.
#'   Backend will keep track of historical values along with timestamps.
#'
#' @param key Name of the metric.
#' @param value Float value for the metric being logged.
#' @param timestamp Unix timestamp in milliseconds at the time metric was logged.
#' @template roxlate-fluent
#'
#' @export
mlflow_log_metric <- function(key, value, timestamp = NULL) {
  active_run <- mlflow_get_or_start_run()
  client <- mlflow_client()
  mlflow_client_log_metric(
    client = client, run_id = run_id(active_run),
    key = key, value = value, timestamp = timestamp
  )
  invisible(value)
}

#' Set Tag
#'
#' Sets a tag on a run. Tags are run metadata that can be updated during and
#'  after a run completes.
#'
#' @param key Name of the tag. Maximum size is 255 bytes. This field is required.
#' @param value String value of the tag being logged. Maximum size is 500 bytes. This field is required.
#' @template roxlate-fluent
#'
#' @export
mlflow_set_tag <- function(key, value) {
  active_run <- mlflow_get_or_start_run()
  client <- mlflow_client()
  mlflow_client_set_tag(
    client = client, run_id = run_id(active_run), key = key, value = value
  )
}

#' End a Run
#'
#' Ends an active MLflow run (if there is one).
#'
#' @param status Updated status of the run. Defaults to `FINISHED`.
#' @template roxlate-fluent
#'
#' @export
mlflow_end_run <- function(status = c("FINISHED", "SCHEDULED", "FAILED", "KILLED")) {
  active_run <- mlflow_active_run()
  if (!is.null(active_run)) {
    status <- match.arg(status)
    client <- mlflow_client()
    mlflow_client_set_terminated(client, run_id(active_run), status)
    mlflow_set_active_run(NULL)
  }
  invisible(NULL)
}

#' Log Parameter
#'
#' Logs a parameter for this run. Examples are params and hyperparams
#'   used for ML training, or constant dates and values used in an ETL pipeline.
#'   A params is a STRING key-value pair. For a run, a single parameter is allowed
#'   to be logged only once.
#'
#' @param key Name of the parameter.
#' @param value String value of the parameter.
#' @template roxlate-fluent
#'
#' @export
mlflow_log_param <- function(key, value) {
  active_run <- mlflow_get_or_start_run()
  client <- mlflow_client()
  mlflow_client_log_param(client, run_id(active_run), key, value)
  invisible(value)
}

#' Log Artifact
#'
#' Logs a specific file or directory as an artifact for this run.
#'
#' @param path The file or directory to log as an artifact.
#' @param artifact_path Destination path within the run's artifact URI.
#' @template roxlate-fluent
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
#'
#' @export
mlflow_log_artifact <- function(path, artifact_path = NULL) {
  active_run <- mlflow_get_or_start_run()
  client <- mlflow_client()
  mlflow_client_log_artifact(client, run_id(active_run), path, artifact_path)
}
