# If one or more external observer(s) are present, then inform them of the
# event. Otherwise do nothing.
mlflow_register_tracking_event <- function(event_name, data) {
  observers <- getOption("MLflowObservers")
  if (length(observers) > 0) {
    lapply(
      observers,
      function(o) {
        o$register_tracking_event(event_name, data)
      }
    )
  }
}
