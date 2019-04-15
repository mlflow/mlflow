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
  mlflow_get_active_run() %||% mlflow_start_run()
}


mlflow_get_experiment_id_from_env <- function(client = mlflow_client()) {
  name <- Sys.getenv("MLFLOW_EXPERIMENT_NAME", unset = NA)
  if (!is.na(name)) {
    mlflow_get_experiment_by_name(client = client, name = name)$experiment_id
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

resolve_client_and_run_id <- function(client, run_id) {
  if (is.null(client)) {
    if (is.null(run_id)) {
      run_id <- mlflow_get_run_id(mlflow_get_or_start_run())
    }
    client <- mlflow_client()
  } else {
    if (is.null(run_id)) stop("`run_id` must be specified when `client` is specified.", call. = FALSE)
  }
  list(client = client, run_id = run_id)
}

parse_run <- function(r) {
  info <- parse_run_info(r$info)

  info$metrics <- parse_run_data(r$data$metrics)
  info$params <- parse_run_data(r$data$params)
  info$tags <- parse_run_data(r$data$tags)

  new_mlflow_run(info)
}

parse_run_info <- function(r) {
  r %>%
    purrr::map_at(c("start_time", "end_time"), milliseconds_to_date) %>%
    tibble::as_tibble()
}

parse_run_data <- function(d) {
  if (is.null(d)) return(NA)
  d %>%
    purrr::transpose() %>%
    purrr::map(unlist) %>%
    purrr::map_at("timestamp", milliseconds_to_date) %>%
    tibble::as_tibble() %>%
    list()
}

resolve_experiment_id <- function(experiment_id) {
  mlflow_infer_experiment_id(experiment_id) %||%
    stop("`experiment_id` must be specified when there is no active experiment.", call. = FALSE)
}

resolve_run_id <- function(run_id) {
  run_id %||%
    mlflow_get_active_run_id() %||%
    stop("`run_id` must be specified when there is no active run.", call. = FALSE)
}

new_mlflow_experiment <- function(x) {
  tibble::new_tibble(x, nrow = 1, class = "mlflow_experiment")
}

new_mlflow_run <- function(x) {
  tibble::new_tibble(x, nrow = 1, class = "mlflow_run")
}
