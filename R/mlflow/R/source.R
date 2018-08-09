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
      message(cnd, "\n")
      mlflow_update_run(status = "FAILED", end_time = current_time())
    },
    interrupt = function(cnd) mlflow_update_run(status = "KILLED", end_time = current_time()),
    {
      suppressPackageStartupMessages(
        source(uri, local = parent.frame())
      )
    }
  )

  invisible(NULL)
}
