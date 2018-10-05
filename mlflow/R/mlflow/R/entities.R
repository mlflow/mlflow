new_mlflow_entities_run <- function(l) {
  run <- purrr::compact(l$run)
  run <- purrr::map_at(run, "info", tidy_run_info)

  structure(
    list(
     info = run$info,
     data = run$data
    ),
    class = "mlflow_run"
  )
}
