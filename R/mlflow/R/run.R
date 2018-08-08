#' Run in MLflow
#'
#' Wrapper for `mlflow run`.
#'
#' @param uri A directory or an R script.
#' @param entry_point Entry point within project, defaults to `main` if not specified.
#' @param param_list A list of parameters.
#' @param experiment_id ID of the experiment under which to launch the run.
#' @param mode Execution mode to use for run.
#' @param cluster_spec Path to JSON file describing the cluster to use when launching a run on Databricks.
#' @param git_username Username for HTTP(S) Git authentication.
#' @param git_password Password for HTTP(S) Git authentication.
#' @param no_conda If specified, assume that MLflow is running within a Conda environment with the necessary
#'   dependencies for the current project instead of attempting to create a new conda environment. Only
#'   valid if running locally.
#' @param storage_dir Only valid when `mode` is local. MLflow downloads artifacts from distributed URIs passed to
#'  parameters of type 'path' to subdirectories of storage_dir.
#' @export
mlflow_run <- function(uri, entry_point = NULL, version = NULL, param_list = NULL,
                       experiment_id = NULL, mode = NULL) {
  param_list <- if (!is.null(param_list)) param_list %>%
    purrr::imap_chr(~ paste0(.y, "=", .x)) %>%
    paste0(collapse = " ")

  args <- list() %>%
    mlflow_cli_param("--entry-point", entry_point) %>%
    mlflow_cli_param("--version", version) %>%
    mlflow_cli_param("--param-list", param_list) %>%
    mlflow_cli_param("--experiment-id", experiment_id) %>%
    mlflow_cli_param("--mode", mode) %>%
    mlflow_cli_param("--cluster_spec", cluster_spec) %>%
    mlflow_cli_param("--git-username", git_username) %>%
    mlflow_cli_param("--git-password", git_password) %>%
    mlflow_cli_param("--no-conda", if (no_conda) "") %>%
    mlflow_cli_param("--storage-dir", storage_dir)

  do.call(mlflow_cli, c("run", args))

  invisible(NULL)
}

#' Source a Script with MLflow Params
#'
#' This function should not be used interactively. It is designed to be called via `Rscript` from
#'   the terminal or through the MLflow CLI.
#'
#' @param uri Path to an R script.
#' @keywords internal
#' @export
mlflow_source <- function(uri) {
  if (interactive()) stop(
    "`mlflow_source()` cannot be used interactively; use `mlflow_run()` instead.",
    call. = FALSE
  )
  .globals$run_params <- list()
  command_args <- parse_command_line(commandArgs(trailingOnly = TRUE))

  if (!is.null(command_args)) {
    purrr::iwalk(command_args, function(value, key) {
      .globals$run_params[[key]] <- value
    })
  }

  tryCatch(
    error = function(cnd) {
      message(cnd)
      mlflow_update_run(status = "FAILED", end_time = current_time())
    },
    interrupt = function(cnd) mlflow_update_run(status = "KILLED", end_time = current_time()),
    {
      source(uri, local = parent.frame())
    }
  )

  invisible(NULL)
}

clear_run <- function() {
  .globals$run_params <- NULL
}



# from rstudio/tfruns R/flags.R
# parse command line arguments
parse_command_line <- function(arguments) {
  if (!length(arguments)) return(NULL)
  # initialize some state
  values <- list()

  i <- 0; n <- length(arguments)
  while (i < n) {
    i <- i + 1
    argument <- arguments[[i]]

    # skip any command line arguments without a '--' prefix
    if (!grepl("^--", argument))
      next

    # terminate if we see "--args" (implies passthrough args)
    if (grepl("^--args$", argument))
      break

    # check to see if an '=' was specified for this argument
    equals_idx <- regexpr("=", argument)
    if (identical(c(equals_idx), -1L)) {
      # no '='; the next argument is the value for this key
      key <- substring(argument, 3)
      val <- arguments[[i + 1]]
      i <- i + 1
    } else {
      # found a '='; the next argument is all the text following
      # that character
      key <- substring(argument, 3, equals_idx - 1)
      val <- substring(argument, equals_idx + 1)
    }

    # convert '-' to '_' in key
    key <- gsub("-", "_", key)

    # update our map of argument values
    values[[key]] <- yaml::yaml.load(val)
  }

  values
}
