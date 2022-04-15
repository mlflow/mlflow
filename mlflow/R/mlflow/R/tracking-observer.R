#' Register an external MLflow observer
#'
#' Registers an external MLflow observer that will receive a
#' `register_tracking_event(event_name, data)` callback on any model tracking
#' event such as "create_run", "delete_run", or "log_metric".
#' Each observer should have a `register_tracking_event(event_name, data)`
#' callback accepting a character vector `event_name` specifying the name of
#' the tracking event, and `data` containing a list of attributes of the event.
#' The callback should be non-blocking, and ideally should complete
#'  instantaneously. Any exception thrown from the callback will be ignored.
#'
#' @examples
#'
#' library(mlflow)
#'
#' observer <- structure(list())
#' observer$register_tracking_event <- function(event_name, data) {
#'   print(event_name)
#'   print(data)
#' }
#' mlflow_register_external_observer(observer)
#'
#' @param observer The observer object (see example)
#' @export
mlflow_register_external_observer <- function(observer) {
  observers <- getOption("MLflowObservers")
  observers <- append(observers, list(observer))
  options(MLflowObservers = observers)
}

# If one or more external observer(s) are present, then inform them of the
# event. Otherwise do nothing.
mlflow_register_tracking_event <- function(event_name, data) {
  observers <- getOption("MLflowObservers")
  if (length(observers) > 0) {
    lapply(
      observers,
      function(o) {
        tryCatch(
          o$register_tracking_event(event_name, data),
          error = function(e) { }
        )
      }
    )
  }
}
