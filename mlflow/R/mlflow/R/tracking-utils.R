mlflow_relative_paths <- function(paths) {
  gsub(paste0("^", file.path(getwd(), "")), "", paths)
}

get_executing_file_name <- function() {
  pattern <- "^--file="
  v <- grep(pattern, commandArgs(), value = TRUE)
  file_name <- gsub(pattern, "", v)
  if (length(file_name)) file_name
}

get_source_name <- function() {
  get_executing_file_name() %||% "<console>"
}

get_source_version <- function() {
  file_name <- get_executing_file_name()
  tryCatch(
    error = function(cnd) NULL,
    {
      repo <- git2r::repository(file_name, discover = TRUE)
      commit <- git2r::commits(repo, n = 1)
      commit[[1]]@sha
    }
  )
}

mlflow_get_or_start_run <- function() {
  mlflow_active_run() %||% mlflow_start_run()
}


mlflow_get_experiment_id_from_env <- function(client = mlflow_client()) {
  name <- Sys.getenv("MLFLOW_EXPERIMENT_NAME", unset = NA)
  if (!is.na(name)) {
    mlflow_client_get_experiment_by_name(client, name)$experiment_id
  } else {
    id <- Sys.getenv("MLFLOW_EXPERIMENT_ID", unset = NA)
    if (is.na(id)) NULL else id
  }
}

mlflow_infer_experiment_id <- function(experiment_id = NULL) {
  experiment_id %||% mlflow_get_active_experiment_id() %||% mlflow_get_experiment_id_from_env()
}

#' @export
with.mlflow_run <- function(data, expr, ...) {

  tryCatch(
    {
      force(expr)
      mlflow_end_run()
    },
    error = function(cnd) {
      message(cnd)
      mlflow_end_run(status = "FAILED")
    },
    interrupt = function(cnd) mlflow_end_run(status = "KILLED")
  )

  invisible(NULL)
}


run_id <- function(run) cast_nullable_string(run$info$run_uuid)

active_run_id <- function() run_id(mlflow_active_run())

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

wait_for <- function(f, wait, sleep) {
  command_start <- Sys.time()

  success <- FALSE
  while (!success && Sys.time() < command_start + wait) {
    success <- suppressWarnings({
      tryCatch({
        f()
        TRUE
      }, error = function(err) {
        FALSE
      })
    })

    if (!success) Sys.sleep(sleep)
  }

  if (!success) {
    stop("Operation failed after waiting for ", wait, " seconds")
  }
}

mlflow_user <- function() {
  if ("user" %in% names(Sys.info()))
    Sys.info()[["user"]]
  else
    "unknown"
}

MLFLOW_SOURCE_TYPE <- list(
  NOTEBOOK = 1,
  JOB = 2,
  PROJECT = 3,
  LOCAL = 4,
  UNKNOWN = 5
)
