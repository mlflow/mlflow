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
  if (!is.null(experiment_id)) mlflow_experiment(experiment_id)

  # Parameter value precedence:
  #   Command line args > `param_list` > MLProject defaults > defaults in script
  .globals$run_params <- list()
  command_args <- parse_command_line(commandArgs(trailingOnly = TRUE))
  passed_params <- config::merge(param_list, command_args)

  # Identify the script to run.
  is_directory <- fs::is_dir(uri)
  script <- if (is_directory) {
    # If `uri` is a directory, check for MLProject.
    if (fs::file_exists(fs::path(uri, "MLProject"))) {
      # MLProject found.
      mlproject <- yaml::yaml.load_file(fs::path(uri, "MLProject"))
      entry_points <- names(mlproject$entry_points)

      if (!is.null(entry_point)) {
        # If `entry_point` is specified, check that it's one of the entry points listed.
        if (!entry_point %in% entry_points)
          stop("Entry point \"" , entry_point, "\" is not found in `MLProject`.", call. = FALSE)
      } else {
        # If no entry point is specified, we go to the sole entry point if it exists.
        if (length(entry_points) == 1) {
          entry_point <- entry_points
        } else {
          # If no entry point is specified, and there are multiple entry points, we default to `main`.
          if (!"main" %in% entry_points)
            stop("`entry_point` must be specified when `MLProject` contains multiple entry points, none of which is \"main\".",
                 call = FALSE)
          entry_point <- "main"
        }
      }

      # Extract parameter defaults from `MLProject` and merge it with `passed_params`.
      passed_params <<- mlproject$entry_points[[entry_point]]$parameters %>%
        purrr::map("default") %>%
        purrr::compact() %>%
        config::merge(passed_params)

      # Return the script path.
      command <- mlproject$entry_points[[entry_point]]$command
      script_path <- regmatches(command, regexpr("(?<=\"|\').*\\.R", command, perl = TRUE))
      if (is.na(script_path))
        stop("Unable to extract script path from entry point entry for \"", entry_point, "\"",
             call. = FALSE)
      if (!fs::file_exists(script_path))
        stop("The file ", script_path, " associated with the entry point ", entry_point, " does not exist.",
             call. = FALSE)
      script_path
    } else {
      # MLProject not found, we check if there's a single R script.
      scripts <- fs::dir_ls(uri, regexp = "\\.R$")
      if (length(scripts) == 1) {
        # If there's a single R script, we'll use that as our entry point.
        scripts[[1]]
      } else {
        # Otherwise, we throw an error.
        stop("There are multiple R scripts in the directory; can't determine which one to execute.",
             call. = FALSE)
      }
    }
  } else {
    # `uri` is a file, so we assume it's the R script to be executed.
    uri
  }

  # Get absolute path to script.
  script <- fs::path_abs(script)

  if (!is.null(passed_params)) {
    purrr::iwalk(passed_params, function(value, key) {
      .globals$run_params[[key]] <- value
    })
  }

  working_dir <- if (is_directory) {
    if (new_dir)
      fs::dir_copy(uri, fs::path_temp())
    else
      uri
  } else
    fs::path_dir(script)

  withr::with_dir(working_dir, {
    source(script, local = parent.frame())
    clear_run()

    # Create dependencies snapshot
    mlflow_snapshot()
  })

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
