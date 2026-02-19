#' Read Command-Line Parameter
#'
#' Reads a command-line parameter passed to an MLflow project
#' MLflow allows you to define named, typed input parameters to your R scripts via the mlflow_param
#' API. This is useful for experimentation, e.g. tracking multiple invocations of the same script
#' with different parameters.
#'
#' @examples
#' \dontrun{
#' # This parametrized script trains a GBM model on the Iris dataset and can be run as an MLflow
#' # project. You can run this script (assuming it's saved at /some/directory/params_example.R)
#' # with custom parameters via:
#' # mlflow_run(entry_point = "params_example.R", uri = "/some/directory",
#' #   parameters = list(num_trees = 200, learning_rate = 0.1))
#' install.packages("gbm")
#' library(mlflow)
#' library(gbm)
#' # define and read input parameters
#' num_trees <- mlflow_param(name = "num_trees", default = 200, type = "integer")
#' lr <- mlflow_param(name = "learning_rate", default = 0.1, type = "numeric")
#' # use params to fit a model
#' ir.adaboost <- gbm(Species ~., data=iris, n.trees=num_trees, shrinkage=lr)
#' }
#'

#' @param name The name of the parameter.
#' @param default The default value of the parameter.
#' @param type Type of this parameter. Required if `default` is not set. If specified,
#'  must be one of "numeric", "integer", or "string".
#' @param description Optional description for the parameter.
#'
#' @export
mlflow_param <- function(name, default = NULL, type = NULL, description = NULL) {
  target_type <- cast_choice(
    type,
    c("numeric", "integer", "string"),
    allow_null = TRUE
  ) %||% typeof(default)

  if (identical(target_type, "NULL"))
    stop("At least one of `default` or `type` must be specified", call. = FALSE)

  caster <- switch(
    target_type,
    integer = cast_nullable_scalar_integer,
    double = cast_nullable_scalar_double,
    numeric = cast_nullable_scalar_double,
    cast_nullable_string
  )

  default <- tryCatch(
    if (!is.null(default)) caster(default),
    error = function(e) stop("`default` value for `", name, "` cannot be casted to type ",
                             type, ": ", conditionMessage(e), call. = FALSE)
  )

  tryCatch(
    caster(.globals$run_params[[name]]) %||% default,
    error = function(e) stop("Provided value for `", name,
                             "` cannot be casted to type ",
                             type, ": ", conditionMessage(e),
                             call. = FALSE)
  )
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
