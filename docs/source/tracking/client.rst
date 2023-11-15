===========================
MLflow Tracking Client APIs
===========================

Some description.

.. _logging-functions:

Logging functions
=================

:py:func:`mlflow.set_tracking_uri` connects to a tracking URI. You can also set the
``MLFLOW_TRACKING_URI`` environment variable to have MLflow find a URI from there. In both cases,
the URI can either be a HTTP/HTTPS URI for a remote server, a database connection string, or a
local path to log data to a directory. The URI defaults to ``mlruns``.

:py:func:`mlflow.get_tracking_uri` returns the current tracking URI.

:py:func:`mlflow.create_experiment` creates a new experiment and returns its ID. Runs can be
launched under the experiment by passing the experiment ID to ``mlflow.start_run``.

:py:func:`mlflow.set_experiment` sets an experiment as active. If the experiment does not exist,
creates a new experiment. If you do not specify an experiment in :py:func:`mlflow.start_run`, new
runs are launched under this experiment.

:py:func:`mlflow.start_run` returns the currently active run (if one exists), or starts a new run
and returns a :py:class:`mlflow.ActiveRun` object usable as a context manager for the
current run. You do not need to call ``start_run`` explicitly: calling one of the logging functions
with no active run automatically starts a new one.

.. note::
  - If the argument ``run_name`` is not set within :py:func:`mlflow.start_run`, a unique run name will be generated for each run.

:py:func:`mlflow.end_run` ends the currently active run, if any, taking an optional run status.

:py:func:`mlflow.active_run` returns a :py:class:`mlflow.entities.Run` object corresponding to the
currently active run, if any.
**Note**: You cannot access currently-active run attributes
(parameters, metrics, etc.) through the run returned by ``mlflow.active_run``. In order to access
such attributes, use the :py:class:`MlflowClient <mlflow.client.MlflowClient>` as follows:

.. code-block:: python

    client = mlflow.MlflowClient()
    data = client.get_run(mlflow.active_run().info.run_id).data

:py:func:`mlflow.last_active_run` returns a :py:class:`mlflow.entities.Run` object corresponding to the
currently active run, if any. Otherwise, it returns a :py:class:`mlflow.entities.Run` object corresponding
the last run started from the current Python process that reached a terminal status (i.e. FINISHED, FAILED, or KILLED).

:py:func:`mlflow.get_parent_run` returns a :py:class:`mlflow.entities.Run` object corresponding to the
parent run for the given run id, if one exists. Otherwise, it returns None.

:py:func:`mlflow.log_param` logs a single key-value param in the currently active run. The key and
value are both strings. Use :py:func:`mlflow.log_params` to log multiple params at once.

:py:func:`mlflow.log_metric` logs a single key-value metric. The value must always be a number.
MLflow remembers the history of values for each metric. Use :py:func:`mlflow.log_metrics` to log
multiple metrics at once.

:py:func:`mlflow.log_input` logs a single :py:class:`mlflow.data.dataset.Dataset` object corresponding to the currently 
active run. You may also log a dataset context string and a dict of key-value tags.

:py:func:`mlflow.set_tag` sets a single key-value tag in the currently active run. The key and
value are both strings. Use :py:func:`mlflow.set_tags` to set multiple tags at once.

:py:func:`mlflow.log_artifact` logs a local file or directory as an artifact, optionally taking an
``artifact_path`` to place it in within the run's artifact URI. Run artifacts can be organized into
directories, so you can place the artifact in a directory this way.

:py:func:`mlflow.log_artifacts` logs all the files in a given directory as artifacts, again taking
an optional ``artifact_path``.

:py:func:`mlflow.get_artifact_uri` returns the URI that artifacts from the current run should be
logged to.





Performance Tracking with Metrics
=================================

You log MLflow metrics with ``log`` methods in the Tracking API. The ``log`` methods support two alternative methods for distinguishing metric values on the x-axis: ``timestamp`` and ``step``.

``timestamp`` is an optional long value that represents the time that the metric was logged. ``timestamp`` defaults to the current time. ``step`` is an optional integer that represents any measurement of training progress (number of training iterations, number of epochs, and so on). ``step`` defaults to 0 and has the following requirements and properties:

- Must be a valid 64-bit integer value.
- Can be negative.
- Can be out of order in successive write calls. For example, (1, 3, 2) is a valid sequence.
- Can have "gaps" in the sequence of values specified in successive write calls. For example, (1, 5, 75, -20) is a valid sequence.

If you specify both a timestamp and a step, metrics are recorded against both axes independently.

Examples
~~~~~~~~

Python
  .. code-block:: python

    with mlflow.start_run():
        for epoch in range(0, 3):
            mlflow.log_metric(key="quality", value=2 * epoch, step=epoch)

Java and Scala
  .. code-block:: java

    MlflowClient client = new MlflowClient();
    RunInfo run = client.createRun();
    for (int epoch = 0; epoch < 3; epoch ++) {
        client.logMetric(run.getRunId(), "quality", 2 * epoch, System.currentTimeMillis(), epoch);
    }


Visualizing Metrics
-------------------

Here is an example plot of the :ref:`quick start tutorial <quickstart-1>` with the step x-axis and two timestamp axes:

.. figure:: ../_static/images/metrics-step.png

  X-axis step

.. figure:: ../_static/images/metrics-time-wall.png

  X-axis wall time - graphs the absolute time each metric was logged

.. figure:: ../_static/images/metrics-time-relative.png

  X-axis relative time - graphs the time relative to the first metric logged, for each run

.. _organizing-runs-in-experiments:

Organizing Runs in Experiments
==============================

MLflow allows you to group runs under experiments, which can be useful for comparing runs intended
to tackle a particular task. You can create experiments using the :ref:`cli` (``mlflow experiments``) or
the :py:func:`mlflow.create_experiment` Python API. You can pass the experiment name for an individual run
using the CLI (for example, ``mlflow run ... --experiment-name [name]``) or the ``MLFLOW_EXPERIMENT_NAME``
environment variable. Alternatively, you can use the experiment ID instead, via the
``--experiment-id`` CLI flag or the ``MLFLOW_EXPERIMENT_ID`` environment variable.

.. code-block:: bash

    # Set the experiment via environment variables
    export MLFLOW_EXPERIMENT_NAME=fraud-detection

    mlflow experiments create --experiment-name fraud-detection

.. code-block:: python

    # Launch a run. The experiment is inferred from the MLFLOW_EXPERIMENT_NAME environment
    # variable, or from the --experiment-name parameter passed to the MLflow CLI (the latter
    # taking precedence)
    with mlflow.start_run():
        mlflow.log_param("a", 1)
        mlflow.log_metric("b", 2)


Managing Experiments and Runs with the Tracking Service API
===========================================================

MLflow provides a more detailed Tracking Service API for managing experiments and runs directly,
which is available through client SDK in the :py:mod:`mlflow.client` module.
This makes it possible to query data about past runs, log additional information about them, create experiments,
add tags to a run, and more.

.. rubric:: Example

.. code-block:: python

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiments = (
        client.search_experiments()
    )  # returns a list of mlflow.entities.Experiment
    run = client.create_run(experiments[0].experiment_id)  # returns mlflow.entities.Run
    client.log_param(run.info.run_id, "hello", "world")
    client.set_terminated(run.info.run_id)

.. _launching-multiple-runs:

Launching Multiple Runs in One Program
======================================
Sometimes you want to launch multiple MLflow runs in the same program: for example, maybe you are
performing a hyperparameter search locally or your experiments are just very fast to run. This is
easy to do because the ``ActiveRun`` object returned by :py:func:`mlflow.start_run` is a Python
`context manager <https://docs.python.org/2.5/whatsnew/pep-343.html>`_. You can "scope" each run to
just one block of code as follows:

.. code-block:: python

   with mlflow.start_run():
       mlflow.log_param("x", 1)
       mlflow.log_metric("y", 2)
       ...

The run remains open throughout the ``with`` statement, and is automatically closed when the
statement exits, even if it exits due to an exception.


.. _add-tags-to-runs:

Adding Tags to Runs
===================

The :py:func:`MlflowClient.set_tag() <mlflow.client.MlflowClient.set_tag>` function lets you add custom tags to runs. A tag can only have a single unique value mapped to it at a time. For example:

.. code-block:: python

  client.set_tag(run.info.run_id, "tag_key", "tag_value")

.. important:: Do not use the prefix ``mlflow.`` (e.g. ``mlflow.note``) for a tag.  This prefix is reserved for use by MLflow. See :ref:`system_tags` for a list of reserved tag keys.


.. _system_tags:

System Tags
===========

You can annotate runs with arbitrary tags. Tag keys that start with ``mlflow.`` are reserved for
internal use. The following tags are set automatically by MLflow, when appropriate:

+-------------------------------+----------------------------------------------------------------------------------------+
| Key                           | Description                                                                            |
+===============================+========================================================================================+
| ``mlflow.note.content``       | A descriptive note about this run. This reserved tag is not set automatically and can  |
|                               | be overridden by the user to include additional information about the run. The content |
|                               | is displayed on the run's page under the Notes section.                                |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.parentRunId``        | The ID of the parent run, if this is a nested run.                                     |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.user``               | Identifier of the user who created the run.                                            |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.source.type``        | Source type. Possible values: ``"NOTEBOOK"``, ``"JOB"``, ``"PROJECT"``,                |
|                               | ``"LOCAL"``, and ``"UNKNOWN"``                                                         |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.source.name``        | Source identifier (e.g., GitHub URL, local Python filename, name of notebook)          |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.source.git.commit``  | Commit hash of the executed code, if in a git repository.                              |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.source.git.branch``  | Name of the branch of the executed code, if in a git repository.                       |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.source.git.repoURL`` | URL that the executed code was cloned from.                                            |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.project.env``        | The runtime context used by the MLflow project.                                        |
|                               | Possible values: ``"docker"`` and ``"conda"``.                                         |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.project.entryPoint`` | Name of the project entry point associated with the current run, if any.               |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.docker.image.name``  | Name of the Docker image used to execute this run.                                     |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.docker.image.id``    | ID of the Docker image used to execute this run.                                       |
+-------------------------------+----------------------------------------------------------------------------------------+
| ``mlflow.log-model.history``  | Model metadata collected by log-model calls. Includes the serialized                   |
|                               | form of the MLModel model files logged to a run, although the exact format and         |
|                               | information captured is subject to change.                                             |
+-------------------------------+----------------------------------------------------------------------------------------+
