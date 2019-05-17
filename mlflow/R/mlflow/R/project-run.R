#' Run an MLflow Project
#'
#' Wrapper for `mlflow run`.
#'
#' @param entry_point Entry point within project, defaults to `main` if not specified.
#' @param uri A directory containing modeling scripts, defaults to the current directory.
#' @param version Version of the project to run, as a Git commit reference for Git projects.
#' @param parameters A list of parameters.
#' @param experiment_id ID of the experiment under which to launch the run.
#' @param experiment_name Name of the experiment under which to launch the run.
#' @param backend Execution backend to use for run.
#' @param backend_config Path to JSON file which will be passed to the backend. For the Databricks backend,
#'   it should describe the cluster to use when launching a run on Databricks.
#' @param no_conda If specified, assume that MLflow is running within a Conda environment with the necessary
#'   dependencies for the current project instead of attempting to create a new Conda environment. Only
#'   valid if running locally.
#' @param storage_dir Valid only when `backend` is local. MLflow downloads artifacts from distributed URIs passed to
#'  parameters of type `path` to subdirectories of `storage_dir`.
#'
#' @return The run associated with this run.
#'
#' @export
mlflow_run <- function(uri = ".", entry_point = NULL, version = NULL, parameters = NULL,
                       experiment_id = NULL, experiment_name = NULL, backend = NULL, backend_config = NULL,
                       no_conda = FALSE, storage_dir = NULL) {
  if (!is.null(experiment_name) && !is.null(experiment_id)) {
    stop("Specify only one of `experiment_name` or `experiment_id`.")
  }
  if (is.null(experiment_name)) {
    experiment_id <- mlflow_infer_experiment_id(experiment_id)
  }
  if (file.exists(uri))
    uri <- fs::path_expand(uri)

  param_list <- if (!is.null(parameters)) parameters %>%
    purrr::imap_chr(~ paste0(.y, "=", .x)) %>%
    purrr::reduce(~ mlflow_cli_param(.x, "--param-list", .y), .init = list())

  args <- list(uri) %>%
    mlflow_cli_param("--entry-point", entry_point) %>%
    mlflow_cli_param("--version", version) %>%
    mlflow_cli_param("--experiment-id", experiment_id) %>%
    mlflow_cli_param("--experiment-name", experiment_name) %>%
    mlflow_cli_param("--backend", backend) %>%
    mlflow_cli_param("--backend-config", backend_config) %>%
    mlflow_cli_param("--storage-dir", storage_dir) %>%
    c(param_list)

  args <- if (!no_conda) args else c(args, "--no-conda")
  result <- do.call(mlflow_cli, c("run", args))
  matches <- regexec(".*Run \\(ID \\'([^\\']+).*", result$stderr)
  run_id <- regmatches(result$stderr, matches)[[1]][[2]]
  invisible(run_id)
}
