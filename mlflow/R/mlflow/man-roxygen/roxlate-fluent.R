#' @details The fluent API family of functions operate with an implied MLflow client
#'   determined by the service set by `mlflow_set_tracking_uri()`. For operations
#'   involving a run it adopts the current active run, or, if one does not exist,
#'   starts one through the implied service.
