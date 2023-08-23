.. _monitoring:

==========
Monitoring
==========

MLflow monitoring capabilities provide insights into the performance and usage of your MLflow server deployment. This documentation outlines the metrics exposed by the MLflow Prometheus exporter.

MLflow relies on the `Prometheus <https://prometheus.io>`_ monitoring system. Prometheus is an open-source system designed for collecting, storing, and querying metrics, making it a popular choice for monitoring dynamic and distributed applications.

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

To activate the exporter, use the ``--expose-prometheus`` option when starting the MLflow server. The option allows you to specify the path to the directory where the metrics will be temporarily stored and shared for multiprocess support. If the directory does not exist, it will be created. The directory must be empty in the server startup. The server will expose metrics to the ``/metrics`` endpoint.

The available metrics are:

.. include:: ../generated/monitoring/metrics.rst

All metrics are grouped by ``mlflow_version`` label.

See also the `CLI docs <cli.html#mlflow-server>`_ for more information.
