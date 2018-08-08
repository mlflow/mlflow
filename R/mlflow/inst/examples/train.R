library(mlflow)

with(mlflow_start_run(), {
  mlflow_log_param("parameter", 0)
  mlflow_log_metric("metric", 0)
})
