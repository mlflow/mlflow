#' Run in MLflow
#'
#' Runs the given file, expression or function within
#' the context of an MLflow run.
#'
#' @param uri A directory or an R script.
#' @param entry_point Entry point within project, defaults to `main` if not specified.
#' @param param_list A list of parameters.
#' @param experiment_id ID of the experiment under which to launch the run.
#' @param new_dir If `TRUE`, copies the project into a temporary working directory and
#'   runs it from there. Otherwise, uses `uri` as the working directory when running the
#'   project.
#'
#' @export
mlflow_run <- function(uri, entry_point = NULL, param_list = NULL,
                       experiment_id = NULL, new_dir = FALSE) {
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
      mlflow_update_run(run_uuid = runid,status = "FAILED", end_time = current_time())
    },
    interrupt = function(cnd) mlflow_update_run(
      run_uuid = runid, status = "KILLED", end_time = current_time()
    ),
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
