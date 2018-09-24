#' Update Run
#'
#' @param run_uuid Unique identifier for the run.
#' @param status Updated status of the run. Defaults to `FINISHED`.
#' @param end_time Unix timestamp of when the run ended in milliseconds.
#' @export
mlflow_update_run <- function(status = c("FINISHED", "SCHEDULED", "FAILED", "KILLED"),
                              end_time = NULL,
                              run_uuid = NULL) {
  mlflow_get_or_create_active_connection()
  run_uuid <- run_uuid %||%
    mlflow_active_run()$run_info$run_uuid %||%
    stop("`run_uuid` must be specified when there is no active run.")

  status <- match.arg(status)
  end_time <- end_time %||% current_time()

  response <- mlflow_rest("runs", "update", verb = "POST", data = list(
    run_uuid = run_uuid,
    status = status,
    end_time = end_time
  ))

  tidy_run_info(response$run_info)
}

current_time <- function() {
  round(as.numeric(Sys.time()) * 1000)
}

milliseconds_to_date <- function(x) as.POSIXct(as.double(x) / 1000, origin = "1970-01-01")

tidy_run_info <- function(run_info) {
  df <- as.data.frame(run_info, stringsAsFactors = FALSE)
  df$start_time <- milliseconds_to_date(df$start_time %||% NA)
  df$end_time <- milliseconds_to_date(df$end_time %||% NA)
  df
}
