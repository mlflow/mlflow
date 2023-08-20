.. _monitoring:

==========
Monitoring
==========

.. contents:: Table of Contents
   :local:
   :depth: 3

MLflow monitoring capabilities provide insights into the performance and usage of your MLflow server deployment. This documentation outlines the metrics and statistics exposed by the MLflow Prometheus exporter.

MLflow relies on the `Prometheus <https://prometheus.io>`_ monitoring system. Prometheus is an open-source system designed for collecting, storing, and querying metrics, making it a popular choice for monitoring dynamic and distributed applications.

Metrics
-------

The server exposes metrics to the `/metrics` endpoint. These metrics provide insights into the behavior and performance of your MLflow server. To activate the exporter, use the `--expose-prometheus` option when starting the MLflow server. The option allows you to specify the path to the directory where metrics will be temporarily stored and shared for multiprocess support. If the directory does not exist, it will be created. The directory must be empty in the server startup.

List of metrics:

.. list-table::
   :widths: 10 10 10
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - mlflow_exporter_info
     - Gauge
     - Information about the Prometheus Flask exporter
   * - mlflow_http_request_total
     - Counter
     - Total number of HTTP requests
   * - mlflow_http_request_exceptions_total
     - Counter
     - Total number of HTTP requests which resulted in an exception
   * - mlflow_http_request_duration_seconds
     - Histogram
     - Flask HTTP request duration in seconds

All metrics are grouped by `mlflow_version` label.

Statistics
----------

The server also provides statistics related to the application. These statistics can help you understand patterns and trends in how MLflow is being used. If enabled, statistics are collected periodically and exposed in the `/metrics` endpoint through the Prometheus exporter.

To enable the collection of statistics, use the `--enable-statistics` option. The server will only expose statistics if the `--expose-prometheus` option is also enabled. You can control the frequency of statistics collection using the `--statistics-update-interval` option, specifying the time interval in seconds. By default, statistics are collected at intervals of one hour (3600 seconds).

List of statistics:

.. list-table::
   :widths: 10 10 10
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - mlflow_user_count
     - Gauge
     - Total number of users
   * - mlflow_experiment_count
     - Gauge
     - Total number of experiments
   * - mlflow_run_count
     - Gauge
     - Total number of runs
   * - mlflow_dataset_count
     - Gauge
     - Total number of datasets
   * - mlflow_registered_model_count
     - Gauge
     - Total number of registered models
   * - mlflow_model_version_count
     - Gauge
     - Total number of model versions

The `registered_model_count` and `model_version_count` statistics include a `stage` label referencing the model stages (e.g. Production).

All statistics are grouped by `mlflow_version` label.
