.. _system-metrics:

System Metrics
==============

MLflow allows users to log system metrics including CPU stats, GPU stats, memory usage, network traffic, and
disk usage during MLflow experiment. In this guide, we will walk through how to manage system metrics logging
with MLflow.

Extra Dependencies
-------------------

To log system metrics in MLflow, please install ``psutil``. We explicitly don't include ``psutil`` in MLflow's
dependencies because ``psutil`` wheel is not available for linux aarch64, and building from source fails intermittently.
To install ``psutil``, run the following command:

.. code-block:: bash

    pip install psutil

If you want to catch GPU metrics, you also need to install ``pynvml``:

.. code-block:: bash

    pip install pynvml

Turn on/off System Metrics Logging
----------------------------------

There are three ways to turn on/off system metrics logging:

- Use environment variable ``MLFLOW_LOG_SYSTEM_METRICS`` to turn on/off system metrics logging for all MLflow runs.
- Use :py:func:`mlflow.enable_system_metrics_logging()` and :py:func:`mlflow.disable_system_metrics_logging()` to turn
  on/off system metrics logging for all MLflow runs.
- Use ``log_system_metrics`` parameter in :py:func:`mlflow.start_run()` to turn on/off system metrics logging for
  the current MLflow run.

Use Environment Variable
~~~~~~~~~~~~~~~~~~~~~~~~

You can set environment variable ``MLFLOW_LOG_SYSTEM_METRICS`` to ``true`` to turn on system metrics logging globally.

.. code-block:: bash

    export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

If you are using Ipython notebook (Jupyter, Databricks notebook, Google Colab), ``export`` command will not work because
of ephemeral shell. Instead you can use the following code:

.. code-block:: python

    import os

    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

Now starting an MLflow run will automatically collect and log the default system metrics. Try running the following code
in your favorite environment, and you should see system metrics existing in the logged run data. Please note that you don't
necessarilty need to start an MLflow server, as the metrics are logged locally.

.. code-block:: python

    import mlflow
    import time

    with mlflow.start_run() as run:
        time.sleep(15)

    print(mlflow.MlflowClient().get_run(run.info.run_id).data)

Your output should look like this:

.. code-block:: output

    <RunData: metrics={'system/cpu_utilization_percentage': 12.4,
    'system/disk_available_megabytes': 213744.0,
    'system/disk_usage_megabytes': 28725.3,
    'system/disk_usage_percentage': 11.8,
    'system/network_receive_megabytes': 0.0,
    'system/network_transmit_megabytes': 0.0,
    'system/system_memory_usage_megabytes': 771.1,
    'system/system_memory_usage_percentage': 5.7}, params={}, tags={'mlflow.runName': 'nimble-auk-61',
    'mlflow.source.name': '/usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py',
    'mlflow.source.type': 'LOCAL',
    'mlflow.user': 'root'}>

To disable system metrics logging, you can use either of the following commands:

.. code-block:: bash

    export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING="false"

.. code-block:: python

    import os

    del os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"]

Rerunning the MLflow code above will not log system metrics.

Use ``mlflow.enable_system_metrics_logging()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide a pair of APIs ``mlflow.enable_system_metrics_logging()`` and
``mlflow.disable_system_metrics_logging()`` to turn on/off system metrics logging globally, since
sometimes you won't be able to set the environment variable. Running the following code will have the
same effect as setting ``MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING`` environment variable to ``true``:

.. code-block:: python

    import mlflow

    mlflow.enable_system_metrics_logging()

    with mlflow.start_run() as run:
        time.sleep(15)

    print(mlflow.MlflowClient().get_run(run.info.run_id).data)

Turn on System Metrics Logging for a Single Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to controlling system metrics logging globally, you can also control it for a
single run. To do so, set ``log_system_metrics`` as ``True`` or ``False`` accordingly in :py:func:`mlflow.start_run()`:

.. code-block:: python

    with mlflow.start_run(log_system_metrics=True) as run:
        time.sleep(15)

    print(mlflow.MlflowClient().get_run(run.info.run_id).data)

The above code will log system metrics for the current run, even if you have disabled system metrics logging by setting
``MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING`` to ``false`` or calling ``mlflow.disable_system_metrics_logging()``.

Type of System Metrics
----------------------

By default MLflow logs the following system metrics:

* cpu_utilization_percentage
* system_memory_usage_megabytes
* system_memory_usage_percentage
* gpu_utilization_percentage
* gpu_memory_usage_megabytes
* gpu_memory_usage_percentage
* network_receive_megabytes
* network_transmit_megabytes
* disk_usage_megabytes
* disk_available_megabytes

GPU metrics are only logged when GPU is available and ``pynvml`` is installed.

Every system metric has a prefix ``system/`` when logged for grouping purpose. So the actual metric name gets logged
are ``system/cpu_utilization_percentage``, ``system/system_memory_usage_megabytes``, etc.

View System Metrics on MLflow UI
--------------------------------

System metrics are available on MLflow UI under the metrics section, let's start our MLflow UI server, and log system
metrics to it:

.. code-block:: bash

    mlflow ui

.. code-block:: python

    import mlflow
    import time

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run() as run:
        time.sleep(15)

Navigate to `http://127.0.0.1:5000 <https://docs.databricks.com/en/machine-learning/index.html>`_ in your browser,
and open your run, you should see system metrics under the metrics section, similar as shown by the screenshot below:

.. figure:: ../_static/images/system-metrics/system-metrics-view.png
    :width: 800px
    :align: center
    :alt: system metrics on MLflow UI


Customize System Metrics Logging
---------------------------------

Customize Logging Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default system metrics are sampled every 10 seconds, and directly logged after sampling. You can customize
the sampling frequency by setting environment variable ``MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL`` or
using :py:func:`mlflow.set_system_metrics_sampling_interval()`. You can also customize the number of samples to aggregate
before logging by setting environment variable ``MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING`` or using
:py:func:`mlflow.set_system_metrics_samples_before_logging()`. The actual logging time window is the product of
``MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL`` and ``MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING``. For example, if
you set sample interval to 1 seconds and samples before logging to 3, then system metrics will be logged every 3s.

.. code-block::python

    import mlflow

    mlflow.set_system_metrics_sampling_interval(1)
    mlflow.set_system_metrics_samples_before_logging(3)

    with mlflow.start_run(log_system_metrics=True) as run:
        time.sleep(15)

    metric_history = mlflow.MlflowClient().get_metric_history(
        run.info.run_id,
        "system/cpu_utilization_percentage",
    )
    print(metric_history)

You will see ``system/cpu_utilization_percentage`` logged for a few times.
