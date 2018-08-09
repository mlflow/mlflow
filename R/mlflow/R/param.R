#' Read Command Line Parameter
#'
#' Reads a command line parameter.
#'
#' @param name The name for this parameter.
#' @param default The default value for this parameter.
#' @param type Type of this parameter. Required if `default` is not set.
#' @param description Optional description for this parameter.
#'
#' @export
mlflow_param <- function(name, default = NULL, type = NULL, description = NULL) {
  if (is.null(default) && is.null(type))
    stop("At least one of `default` or `type` must be specified", call. = FALSE)
  if (!is.null(default) && !is.null(type) && !identical(typeof(default), type))
    stop("`default` value is not of type ", type, ".", call. = FALSE)
  .globals$run_params[[name]] %||% default
}

# from rstudio/tfruns R/flags.R
# parse command line arguments
parse_command_line <- function(arguments) {
  arguments <- arguments[!arguments == "--args"]

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
