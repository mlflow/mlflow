.. _monitoring:

==========
Monitoring
==========

MLflow monitoring capabilities provide insights into the performance and usage of your MLflow server deployment. This documentation outlines the metrics and statistics exposed by the MLflow Prometheus exporter.

MLflow relies on the `Prometheus <https://prometheus.io>`_ monitoring system. Prometheus is an open-source system designed for collecting, storing, and querying metrics, making it a popular choice for monitoring dynamic and distributed applications.

See also the `CLI docs <cli.html#mlflow-server>`_ for more information.

.. contents:: Table of Contents
   :local:
   :depth: 3

Installation
------------

Install MLflow with the Prometheus exporter feature:

.. code-block:: bash

    pip install mlflow[prometheus]

Metrics
-------

The server exposes metrics to the ``/metrics`` endpoint. These metrics provide insights into the behavior and performance of your MLflow server. To activate the exporter, use the ``--expose-prometheus`` option when starting the MLflow server. The option allows you to specify the path to the directory where metrics will be temporarily stored and shared for multiprocess support. If the directory does not exist, it will be created. The directory must be empty in the server startup.

The available metrics are:

.. include:: ../generated/monitoring/metrics.rst

All metrics are grouped by ``mlflow_version`` label.

Statistics
----------

The server also provides statistics related to the application. These statistics can help you understand patterns and trends in how MLflow is being used. If enabled, statistics are collected periodically and exposed in the ``/metrics`` endpoint through the Prometheus exporter.

To enable the collection of statistics, use the ``--enable-statistics`` option. The server will only expose statistics if the ``--expose-prometheus`` option is also enabled. You can control the frequency of statistics collection using the ``--statistics-update-interval`` option, specifying the time interval in seconds. By default, statistics are collected at intervals of one hour (3600 seconds).

The available statistics are:

.. include:: ../generated/monitoring/statistics.rst

The ``registered_model_count`` and ``model_version_count`` statistics include a ``stage`` label referencing the model stages (e.g. **Production**).

All statistics are grouped by ``mlflow_version`` label.
