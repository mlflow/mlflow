#' Run in MLflow
#'
#' Runs the given file, expression or function within
#' the context of an MLflow run.
#'
#' @param x A directory or an R script.
#' @param params A list of parameters.
#'
#' @export
mlflow_run <- function(x, params = NULL) {
  # Suppose `x` is a script for now.

  # Create dependencies snapshot
  mlflow_snapshot()

  .globals$run_params <- list()
  command_args <- parse_command_line(commandArgs(trailingOnly = TRUE))

  # Precedence: Command line args > `params` > defaults in script
  passed_params <- config::merge(params, command_args)

  if (!is.null(passed_params)) {
    purrr::iwalk(passed_params, function(value, key) {
      .globals$run_params[[key]] <- value
    })
  }

  source(x, local = parent.frame())
  clear_run()
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
