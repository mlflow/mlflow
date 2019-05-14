.. _R-api:

========
R API
========

The MLflow R API allows you to use MLflow :doc:`Tracking <tracking/>`, :doc:`Projects <projects/>` and :doc:`Models <models/>`.

You can use the R API to `install MLflow`_, start the `user interface <Run MLflow user interface_>`_, `create <Create Experiment_>`_ and `list experiments <List Experiments_>`_, `save models <Save Model for MLflow_>`_, `run projects <Run in MLflow_>`_ and `serve models <Serve an RFunc MLflow Model_>`_ among many other functions available in the R API.

.. contents:: Table of Contents
    :local:
    :depth: 1

Initialize an MLflow Client
===========================

Initialize an MLflow Client

.. code:: r

   mlflow_client(tracking_uri = NULL)

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``tracking_uri``              | The tracking URI. If not provided,   |
|                               | defaults to the service set by       |
|                               | ``mlflow_set_tracking_uri()``.       |
+-------------------------------+--------------------------------------+

Create Experiment
=================

Creates an MLflow experiment and returns its id.

.. code:: r

   mlflow_create_experiment(name, artifact_location = NULL, client = NULL)

.. _arguments-1:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``name``                      | The name of the experiment to        |
|                               | create.                              |
+-------------------------------+--------------------------------------+
| ``artifact_location``         | Location where all artifacts for     |
|                               | this experiment are stored. If not   |
|                               | provided, the remote server will     |
|                               | select an appropriate default.       |
+-------------------------------+--------------------------------------+
| ``client``                    | (Optional) An ``mlflow_client``      |
|                               | object.                              |
+-------------------------------+--------------------------------------+

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Delete Experiment
=================

Marks an experiment and associated runs, params, metrics, etc. for
deletion. If the experiment uses FileStore, artifacts associated with
experiment are also deleted.

.. code:: r

   mlflow_delete_experiment(experiment_id, client = NULL)

.. _arguments-2:

Arguments
---------

+-----------------------------------+-----------------------------------+
| Argument                          | Description                       |
+===================================+===================================+
| ``experiment_id``                 | ID of the associated experiment.  |
|                                   | This field is required.           |
+-----------------------------------+-----------------------------------+
| ``client``                        | (Optional) An ``mlflow_client``   |
|                                   | object.                           |
+-----------------------------------+-----------------------------------+

.. _details-1:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Delete a Run
============

Delete a Run

.. code:: r

   mlflow_delete_run(run_id, client = NULL)

.. _arguments-3:

Arguments
---------

+------------+-----------------------------------------+
| Argument   | Description                             |
+============+=========================================+
| ``run_id`` | Run ID.                                 |
+------------+-----------------------------------------+
| ``client`` | (Optional) An ``mlflow_client`` object. |
+------------+-----------------------------------------+

.. _details-2:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Download Artifacts
==================

Download an artifact file or directory from a run to a local directory
if applicable, and return a local path for it.

.. code:: r

   mlflow_download_artifacts(path, run_id = NULL, client = NULL)

.. _arguments-4:

Arguments
---------

+------------+-----------------------------------------------+
| Argument   | Description                                   |
+============+===============================================+
| ``path``   | Relative source path to the desired artifact. |
+------------+-----------------------------------------------+
| ``run_id`` | Run ID.                                       |
+------------+-----------------------------------------------+
| ``client`` | (Optional) An ``mlflow_client`` object.       |
+------------+-----------------------------------------------+

.. _details-3:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

End a Run
=========

Terminates a run. Attempts to end the current active run if ``run_id``
is not specified.

.. code:: r

   mlflow_end_run(status = c("FINISHED", "SCHEDULED", "FAILED", "KILLED"),
     end_time = NULL, run_id = NULL, client = NULL)

.. _arguments-5:

Arguments
---------

+--------------+-------------------------------------------------------+
| Argument     | Description                                           |
+==============+=======================================================+
| ``status``   | Updated status of the run. Defaults to ``FINISHED``.  |
+--------------+-------------------------------------------------------+
| ``end_time`` | Unix timestamp of when the run ended in milliseconds. |
+--------------+-------------------------------------------------------+
| ``run_id``   | Run ID.                                               |
+--------------+-------------------------------------------------------+
| ``client``   | (Optional) An ``mlflow_client`` object.               |
+--------------+-------------------------------------------------------+

.. _details-4:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Get Experiment
==============

Gets metadata for an experiment and a list of runs for the experiment.
Attempts to obtain the active experiment if both ``experiment_id`` and
``name`` are unspecified.

.. code:: r

   mlflow_get_experiment(experiment_id = NULL, name = NULL,
     client = NULL)

.. _arguments-6:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``experiment_id``             | Identifer to get an experiment.      |
+-------------------------------+--------------------------------------+
| ``name``                      | The experiment name. Only one of     |
|                               | ``name`` or ``experiment_id`` should |
|                               | be specified.                        |
+-------------------------------+--------------------------------------+
| ``client``                    | (Optional) An ``mlflow_client``      |
|                               | object.                              |
+-------------------------------+--------------------------------------+

.. _details-5:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Get Metric History
==================

Get a list of all values for the specified metric for a given run.

.. code:: r

   mlflow_get_metric_history(metric_key, run_id = NULL, client = NULL)

.. _arguments-7:

Arguments
---------

+----------------+-----------------------------------------+
| Argument       | Description                             |
+================+=========================================+
| ``metric_key`` | Name of the metric.                     |
+----------------+-----------------------------------------+
| ``run_id``     | Run ID.                                 |
+----------------+-----------------------------------------+
| ``client``     | (Optional) An ``mlflow_client`` object. |
+----------------+-----------------------------------------+

.. _details-6:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Get Run
=======

Gets metadata, params, tags, and metrics for a run. In the case where
multiple metrics with the same key are logged for the run, returns only
the value with the latest timestamp. If there are multiple values with
the latest timestamp, returns the maximum of these values.

.. code:: r

   mlflow_get_run(run_id = NULL, client = NULL)

.. _arguments-8:

Arguments
---------

+------------+-----------------------------------------+
| Argument   | Description                             |
+============+=========================================+
| ``run_id`` | Run ID.                                 |
+------------+-----------------------------------------+
| ``client`` | (Optional) An ``mlflow_client`` object. |
+------------+-----------------------------------------+

.. _details-7:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Get Remote Tracking URI
=======================

Gets the remote tracking URI.

.. code:: r

   mlflow_get_tracking_uri()

Get Run or Experiment ID
========================

Extracts the ID of the run or experiment.

.. code:: r

   mlflow_id(object)
   list(list("mlflow_id"), list("mlflow_run"))(object)
   list(list("mlflow_id"), list("mlflow_experiment"))(object)

.. _arguments-9:

Arguments
---------

+------------+----------------------------------------------------+
| Argument   | Description                                        |
+============+====================================================+
| ``object`` | An ``mlflow_run`` or ``mlflow_experiment`` object. |
+------------+----------------------------------------------------+

Install MLflow
==============

Installs MLflow for individual use.

.. code:: r

   mlflow_install()

.. _details-8:

Details
-------

MLflow requires Python and Conda to be installed. See
https://www.python.org/getit/ and
https://docs.conda.io/projects/conda/en/latest/user-guide/install/ .

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "mlflow_install()\n") 
    

List Artifacts
==============

Gets a list of artifacts.

.. code:: r

   mlflow_list_artifacts(path = NULL, run_id = NULL, client = NULL)

.. _arguments-10:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``path``                      | The run’s relative artifact path to  |
|                               | list from. If not specified, it is   |
|                               | set to the root artifact path        |
+-------------------------------+--------------------------------------+
| ``run_id``                    | Run ID.                              |
+-------------------------------+--------------------------------------+
| ``client``                    | (Optional) An ``mlflow_client``      |
|                               | object.                              |
+-------------------------------+--------------------------------------+

.. _details-9:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

List Experiments
================

Gets a list of all experiments.

.. code:: r

   mlflow_list_experiments(view_type = c("ACTIVE_ONLY", "DELETED_ONLY",
     "ALL"), client = NULL)

.. _arguments-11:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``view_type``                 | Qualifier for type of experiments to |
|                               | be returned. Defaults to             |
|                               | ``ACTIVE_ONLY``.                     |
+-------------------------------+--------------------------------------+
| ``client``                    | (Optional) An ``mlflow_client``      |
|                               | object.                              |
+-------------------------------+--------------------------------------+

.. _details-10:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

List Run Infos
==============

List run infos.

.. code:: r

   mlflow_list_run_infos(run_view_type = c("ACTIVE_ONLY", "DELETED_ONLY",
     "ALL"), experiment_id = NULL, client = NULL)

.. _arguments-12:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``run_view_type``             | Run view type.                       |
+-------------------------------+--------------------------------------+
| ``experiment_id``             | Experiment ID. Attempts to use the   |
|                               | active experiment if not specified.  |
+-------------------------------+--------------------------------------+
| ``client``                    | (Optional) An ``mlflow_client``      |
|                               | object.                              |
+-------------------------------+--------------------------------------+

.. _details-11:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Load MLflow Model Flavor
========================

Loads an MLflow model flavor, to be used by package authors to extend
the supported MLflow models.

.. code:: r

   mlflow_load_flavor(model_path)

.. _arguments-13:

Arguments
---------

+----------------+------------------------------------------------------------+
| Argument       | Description                                                |
+================+============================================================+
| ``model_path`` | The path to the MLflow model wrapped in the correct class. |
+----------------+------------------------------------------------------------+

Load MLflow Model
=================

Loads an MLflow model. MLflow models can have multiple model flavors.
Not all flavors / models can be loaded in R. This method by default
searches for a flavor supported by R/MLflow.

.. code:: r

   mlflow_load_model(model_uri, flavor = NULL, client = mlflow_client())

.. _arguments-14:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``model_uri``                 | The location, in URI format, of the  |
|                               | MLflow model.                        |
+-------------------------------+--------------------------------------+
| ``flavor``                    | Optional flavor specification. Can   |
|                               | be used to load a particular flavor  |
|                               | in case there are multiple flavors   |
|                               | available.                           |
+-------------------------------+--------------------------------------+

.. _details-12:

Details
-------

The URI scheme must be supported by MLflow - i.e. there has to be an
MLflow artifact repository corresponding to the scheme of the URI. The
content is expected to point to a directory containing MLmodel. The
following are examples of valid model uris: -
``file:///absolute/path/to/local/model`` -
``file:relative/path/to/local/model`` - ``s3://my_bucket/path/to/model``
- ``runs:/<mlflow_run_id>/run-relative/path/to/model`` For more
information about supported URI schemes, see the Artifacts Documentation
``<https://www.mlflow.org/docs/latest/tracking.html#supported-artifact-stores>``\ \_.

Seealso
-------

Other artifact uri:
```mlflow_rfunc_predict`` <mlflow_rfunc_predict.html>`__ ,
```mlflow_rfunc_serve`` <mlflow_rfunc_serve.html>`__

Log Artifact
============

Logs a specific file or directory as an artifact for a run.

.. code:: r

   mlflow_log_artifact(path, artifact_path = NULL, run_id = NULL,
     client = NULL)

.. _arguments-15:

Arguments
---------

+-------------------+-------------------------------------------------+
| Argument          | Description                                     |
+===================+=================================================+
| ``path``          | The file or directory to log as an artifact.    |
+-------------------+-------------------------------------------------+
| ``artifact_path`` | Destination path within the run’s artifact URI. |
+-------------------+-------------------------------------------------+
| ``run_id``        | Run ID.                                         |
+-------------------+-------------------------------------------------+
| ``client``        | (Optional) An ``mlflow_client`` object.         |
+-------------------+-------------------------------------------------+

.. _details-13:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

When logging to Amazon S3, ensure that the user has a proper policy
attached to it, for instance:

\`\`

Additionally, at least the ``AWS_ACCESS_KEY_ID`` and
``AWS_SECRET_ACCESS_KEY`` environment variables must be set to the
corresponding key and secrets provided by Amazon IAM.

Log Batch
=========

Log a batch of metrics, params, and/or tags for a run. The server will
respond with an error (non-200 status code) if any data failed to be
persisted. In case of error (due to internal server error or an invalid
request), partial data may be written.

.. code:: r

   mlflow_log_batch(metrics = NULL, params = NULL, tags = NULL,
     run_id = NULL, client = NULL)

.. _arguments-16:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``metrics``                   | A dataframe of metrics to log,       |
|                               | containing the following columns:    |
|                               | “key”, “value”, “step”, “timestamp”. |
|                               | This dataframe cannot contain any    |
|                               | missing (‘NA’) entries.              |
+-------------------------------+--------------------------------------+
| ``params``                    | A dataframe of params to log,        |
|                               | containing the following columns:    |
|                               | “key”, “value”. This dataframe       |
|                               | cannot contain any missing (‘NA’)    |
|                               | entries.                             |
+-------------------------------+--------------------------------------+
| ``tags``                      | A dataframe of tags to log,          |
|                               | containing the following columns:    |
|                               | “key”, “value”. This dataframe       |
|                               | cannot contain any missing (‘NA’)    |
|                               | entries.                             |
+-------------------------------+--------------------------------------+
| ``run_id``                    | Run ID.                              |
+-------------------------------+--------------------------------------+
| ``client``                    | (Optional) An ``mlflow_client``      |
|                               | object.                              |
+-------------------------------+--------------------------------------+

.. _details-14:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Log Metric
==========

Logs a metric for a run. Metrics key-value pair that records a single
float measure. During a single execution of a run, a particular metric
can be logged several times. The MLflow Backend keeps track of
historical metric values along two axes: timestamp and step.

.. code:: r

   mlflow_log_metric(key, value, timestamp = NULL, step = NULL,
     run_id = NULL, client = NULL)

.. _arguments-17:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``key``                       | Name of the metric.                  |
+-------------------------------+--------------------------------------+
| ``value``                     | Float value for the metric being     |
|                               | logged.                              |
+-------------------------------+--------------------------------------+
| ``timestamp``                 | Timestamp at which to log the        |
|                               | metric. Timestamp is rounded to the  |
|                               | nearest integer. If unspecified, the |
|                               | number of milliseconds since the     |
|                               | Unix epoch is used.                  |
+-------------------------------+--------------------------------------+
| ``step``                      | Step at which to log the metric.     |
|                               | Step is rounded to the nearest       |
|                               | integer. If unspecified, the default |
|                               | value of zero is used.               |
+-------------------------------+--------------------------------------+
| ``run_id``                    | Run ID.                              |
+-------------------------------+--------------------------------------+
| ``client``                    | (Optional) An ``mlflow_client``      |
|                               | object.                              |
+-------------------------------+--------------------------------------+

.. _details-15:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Log Model
=========

Logs a model for this run. Similar to ``mlflow_save_model()`` but stores
model as an artifact within the active run.

.. code:: r

   mlflow_log_model(fn, artifact_path)

.. _arguments-18:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``fn``                        | The serving function that will       |
|                               | perform a prediction.                |
+-------------------------------+--------------------------------------+
| ``artifact_path``             | Destination path where this MLflow   |
|                               | compatible model will be saved.      |
+-------------------------------+--------------------------------------+

Log Parameter
=============

Logs a parameter for a run. Examples are params and hyperparams used for
ML training, or constant dates and values used in an ETL pipeline. A
param is a STRING key-value pair. For a run, a single parameter is
allowed to be logged only once.

.. code:: r

   mlflow_log_param(key, value, run_id = NULL, client = NULL)

.. _arguments-19:

Arguments
---------

+------------+-----------------------------------------+
| Argument   | Description                             |
+============+=========================================+
| ``key``    | Name of the parameter.                  |
+------------+-----------------------------------------+
| ``value``  | String value of the parameter.          |
+------------+-----------------------------------------+
| ``run_id`` | Run ID.                                 |
+------------+-----------------------------------------+
| ``client`` | (Optional) An ``mlflow_client`` object. |
+------------+-----------------------------------------+

.. _details-16:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Read Command-Line Parameter
===========================

Reads a command-line parameter.

.. code:: r

   mlflow_param(name, default = NULL, type = NULL, description = NULL)

.. _arguments-20:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``name``                      | The name of the parameter.           |
+-------------------------------+--------------------------------------+
| ``default``                   | The default value of the parameter.  |
+-------------------------------+--------------------------------------+
| ``type``                      | Type of this parameter. Required if  |
|                               | ``default`` is not set. If           |
|                               | specified, must be one of “numeric”, |
|                               | “integer”, or “string”.              |
+-------------------------------+--------------------------------------+
| ``description``               | Optional description for the         |
|                               | parameter.                           |
+-------------------------------+--------------------------------------+

Predict over MLflow Model Flavor
================================

Performs prediction over a model loaded using ``mlflow_load_model()`` ,
to be used by package authors to extend the supported MLflow models.

.. code:: r

   mlflow_predict_flavor(model, data)

.. _arguments-21:

Arguments
---------

+-----------+----------------------------------+
| Argument  | Description                      |
+===========+==================================+
| ``model`` | The loaded MLflow model flavor.  |
+-----------+----------------------------------+
| ``data``  | A data frame to perform scoring. |
+-----------+----------------------------------+

Generate Prediction with MLflow Model
=====================================

Generates a prediction with an MLflow model.

.. code:: r

   mlflow_predict_model(model, data)

.. _arguments-22:

Arguments
---------

+-----------+-------------------------+
| Argument  | Description             |
+===========+=========================+
| ``model`` | MLflow model.           |
+-----------+-------------------------+
| ``data``  | Dataframe to be scored. |
+-----------+-------------------------+

Rename Experiment
=================

Renames an experiment.

.. code:: r

   mlflow_rename_experiment(new_name, experiment_id = NULL, client = NULL)

.. _arguments-23:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``new_name``                  | The experiment’s name will be        |
|                               | changed to this. The new name must   |
|                               | be unique.                           |
+-------------------------------+--------------------------------------+
| ``experiment_id``             | ID of the associated experiment.     |
|                               | This field is required.              |
+-------------------------------+--------------------------------------+
| ``client``                    | (Optional) An ``mlflow_client``      |
|                               | object.                              |
+-------------------------------+--------------------------------------+

.. _details-17:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Restore Experiment
==================

Restores an experiment marked for deletion. This also restores
associated metadata, runs, metrics, and params. If experiment uses
FileStore, underlying artifacts associated with experiment are also
restored.

.. code:: r

   mlflow_restore_experiment(experiment_id, client = NULL)

.. _arguments-24:

Arguments
---------

+-----------------------------------+-----------------------------------+
| Argument                          | Description                       |
+===================================+===================================+
| ``experiment_id``                 | ID of the associated experiment.  |
|                                   | This field is required.           |
+-----------------------------------+-----------------------------------+
| ``client``                        | (Optional) An ``mlflow_client``   |
|                                   | object.                           |
+-----------------------------------+-----------------------------------+

.. _details-18:

Details
-------

Throws ``RESOURCE_DOES_NOT_EXIST`` if the experiment was never created
or was permanently deleted.

When ``client`` is not specified, these functions attempt to infer the
current active client.

Restore a Run
=============

Restore a Run

.. code:: r

   mlflow_restore_run(run_id, client = NULL)

.. _arguments-25:

Arguments
---------

+------------+-----------------------------------------+
| Argument   | Description                             |
+============+=========================================+
| ``run_id`` | Run ID.                                 |
+------------+-----------------------------------------+
| ``client`` | (Optional) An ``mlflow_client`` object. |
+------------+-----------------------------------------+

.. _details-19:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Restore Snapshot
================

Restores a snapshot of all dependencies required to run the files in the
current directory.

.. code:: r

   mlflow_restore_snapshot()

Predict using RFunc MLflow Model
================================

Performs prediction using an RFunc MLflow model from a file or data
frame.

.. code:: r

   mlflow_rfunc_predict(model_uri, input_path = NULL, output_path = NULL,
     data = NULL, restore = FALSE)

.. _arguments-26:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``model_uri``                 | The location, in URI format, of the  |
|                               | MLflow model.                        |
+-------------------------------+--------------------------------------+
| ``input_path``                | Path to ‘JSON’ or ‘CSV’ file to be   |
|                               | used for prediction.                 |
+-------------------------------+--------------------------------------+
| ``output_path``               | ‘JSON’ or ‘CSV’ file where the       |
|                               | prediction will be written to.       |
+-------------------------------+--------------------------------------+
| ``data``                      | Data frame to be scored. This can be |
|                               | used for testing purposes and can    |
|                               | only be specified when               |
|                               | ``input_path`` is not specified.     |
+-------------------------------+--------------------------------------+
| ``restore``                   | Should ``mlflow_restore_snapshot()`` |
|                               | be called before serving?            |
+-------------------------------+--------------------------------------+

.. _details-20:

Details
-------

The URI scheme must be supported by MLflow - i.e. there has to be an
MLflow artifact repository corresponding to the scheme of the URI. The
content is expected to point to a directory containing MLmodel. The
following are examples of valid model uris: -
``file:///absolute/path/to/local/model`` -
``file:relative/path/to/local/model`` - ``s3://my_bucket/path/to/model``
- ``runs:/<mlflow_run_id>/run-relative/path/to/model`` For more
information about supported URI schemes, see the Artifacts Documentation
``<https://www.mlflow.org/docs/latest/tracking.html#supported-artifact-stores>``\ \_.

.. _seealso-1:

Seealso
-------

Other artifact uri: ```mlflow_load_model`` <mlflow_load_model.html>`__ ,
```mlflow_rfunc_serve`` <mlflow_rfunc_serve.html>`__

.. _examples-1:

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "\n", "# save simple model which roundtrips data as prediction\n", "mlflow_save_model(function(df) df, \"mlflow_roundtrip\")\n", "\n", "# save data as json\n", "jsonlite::write_json(iris, \"iris.json\")\n", "\n", "# predict existing model from json data\n", "# load the model from local relative path.\n", "mlflow_rfunc_predict(\"file:mlflow_roundtrip\", \"iris.json\")\n") 
    

Serve an RFunc MLflow Model
===========================

Serves an RFunc MLflow model as a local web API.

.. code:: r

   mlflow_rfunc_serve(model_uri, host = "127.0.0.1", port = 8090,
     daemonized = FALSE, browse = !daemonized, restore = FALSE)

.. _arguments-27:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``model_uri``                 | The location, in URI format, of the  |
|                               | MLflow model.                        |
+-------------------------------+--------------------------------------+
| ``host``                      | Address to use to serve model, as a  |
|                               | string.                              |
+-------------------------------+--------------------------------------+
| ``port``                      | Port to use to serve model, as       |
|                               | numeric.                             |
+-------------------------------+--------------------------------------+
| ``daemonized``                | Makes ``httpuv`` server daemonized   |
|                               | so R interactive sessions are not    |
|                               | blocked to handle requests. To       |
|                               | terminate a daemonized server, call  |
|                               | ``httpuv::stopDaemonizedServer()``   |
|                               | with the handle returned from this   |
|                               | call.                                |
+-------------------------------+--------------------------------------+
| ``browse``                    | Launch browser with serving landing  |
|                               | page?                                |
+-------------------------------+--------------------------------------+
| ``restore``                   | Should ``mlflow_restore_snapshot()`` |
|                               | be called before serving?            |
+-------------------------------+--------------------------------------+

.. _details-21:

Details
-------

The URI scheme must be supported by MLflow - i.e. there has to be an
MLflow artifact repository corresponding to the scheme of the URI. The
content is expected to point to a directory containing MLmodel. The
following are examples of valid model uris: -
``file:///absolute/path/to/local/model`` -
``file:relative/path/to/local/model`` - ``s3://my_bucket/path/to/model``
- ``runs:/<mlflow_run_id>/run-relative/path/to/model`` For more
information about supported URI schemes, see the Artifacts Documentation
``<https://www.mlflow.org/docs/latest/tracking.html#supported-artifact-stores>``\ \_.

.. _seealso-2:

Seealso
-------

Other artifact uri: ```mlflow_load_model`` <mlflow_load_model.html>`__ ,
```mlflow_rfunc_predict`` <mlflow_rfunc_predict.html>`__

.. _examples-2:

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "\n", "# save simple model with constant prediction\n", "mlflow_save_model(function(df) 1, \"mlflow_constant\")\n", "\n", "# serve an existing model over a web interface\n", "mlflow_rfunc_serve(\"mlflow_constant\")\n", "\n", "# request prediction from server\n", "httr::POST(\"http://127.0.0.1:8090/predict/\")\n") 

Run an MLflow Project
=====================

Wrapper for ``mlflow run``.

.. code:: r

   mlflow_run(entry_point = NULL, uri = ".", version = NULL,
     param_list = NULL, experiment_id = NULL, experiment_name = NULL,
     backend = NULL, backend_config = NULL, no_conda = FALSE,
     storage_dir = NULL)

.. _arguments-28:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``entry_point``               | Entry point within project, defaults |
|                               | to ``main`` if not specified.        |
+-------------------------------+--------------------------------------+
| ``uri``                       | A directory containing modeling      |
|                               | scripts, defaults to the current     |
|                               | directory.                           |
+-------------------------------+--------------------------------------+
| ``version``                   | Version of the project to run, as a  |
|                               | Git commit reference for Git         |
|                               | projects.                            |
+-------------------------------+--------------------------------------+
| ``param_list``                | A list of parameters.                |
+-------------------------------+--------------------------------------+
| ``experiment_id``             | ID of the experiment under which to  |
|                               | launch the run.                      |
+-------------------------------+--------------------------------------+
| ``experiment_name``           | Name of the experiment under which   |
|                               | to launch the run.                   |
+-------------------------------+--------------------------------------+
| ``backend``                   | Execution backend to use for run.    |
+-------------------------------+--------------------------------------+
| ``backend_config``            | Path to JSON file which will be      |
|                               | passed to the backend. For the       |
|                               | Databricks backend, it should        |
|                               | describe the cluster to use when     |
|                               | launching a run on Databricks.       |
+-------------------------------+--------------------------------------+
| ``no_conda``                  | If specified, assume that MLflow is  |
|                               | running within a Conda environment   |
|                               | with the necessary dependencies for  |
|                               | the current project instead of       |
|                               | attempting to create a new Conda     |
|                               | environment. Only valid if running   |
|                               | locally.                             |
+-------------------------------+--------------------------------------+
| ``storage_dir``               | Valid only when ``backend`` is       |
|                               | local. MLflow downloads artifacts    |
|                               | from distributed URIs passed to      |
|                               | parameters of type ``path`` to       |
|                               | subdirectories of ``storage_dir``.   |
+-------------------------------+--------------------------------------+

Value
-----

The run associated with this run.

Save MLflow Keras Model Flavor
==============================

Saves model in MLflow Keras flavor.

.. code:: r

   list(list("mlflow_save_flavor"), list("keras.engine.training.Model"))(x,
     path = "model", r_dependencies = NULL, conda_env = NULL)

.. _arguments-29:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``x``                         | The serving function or model that   |
|                               | will perform a prediction.           |
+-------------------------------+--------------------------------------+
| ``path``                      | Destination path where this MLflow   |
|                               | compatible model will be saved.      |
+-------------------------------+--------------------------------------+
| ``r_dependencies``            | Optional vector of paths to          |
|                               | dependency files to include in the   |
|                               | model, as in ``r-dependencies.txt``  |
|                               | or ``conda.yaml`` .                  |
+-------------------------------+--------------------------------------+
| ``conda_env``                 | Path to Conda dependencies file.     |
+-------------------------------+--------------------------------------+

.. _value-1:

Value
-----

This function must return a list of flavors that conform to the MLmodel
specification.

Save MLflow Model Flavor
========================

Saves model in MLflow flavor, to be used by package authors to extend
the supported MLflow models.

.. code:: r

   mlflow_save_flavor(x, path = "model", r_dependencies = NULL,
     conda_env = NULL)

.. _arguments-30:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``x``                         | The serving function or model that   |
|                               | will perform a prediction.           |
+-------------------------------+--------------------------------------+
| ``path``                      | Destination path where this MLflow   |
|                               | compatible model will be saved.      |
+-------------------------------+--------------------------------------+
| ``r_dependencies``            | Optional vector of paths to          |
|                               | dependency files to include in the   |
|                               | model, as in ``r-dependencies.txt``  |
|                               | or ``conda.yaml`` .                  |
+-------------------------------+--------------------------------------+
| ``conda_env``                 | Path to Conda dependencies file.     |
+-------------------------------+--------------------------------------+

.. _value-2:

Value
-----

This function must return a list of flavors that conform to the MLmodel
specification.

Save Model for MLflow
=====================

Saves model in MLflow format that can later be used for prediction and
serving.

.. code:: r

   mlflow_save_model(x, path = "model", r_dependencies = NULL,
     conda_env = NULL)

.. _arguments-31:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``x``                         | The serving function or model that   |
|                               | will perform a prediction.           |
+-------------------------------+--------------------------------------+
| ``path``                      | Destination path where this MLflow   |
|                               | compatible model will be saved.      |
+-------------------------------+--------------------------------------+
| ``r_dependencies``            | Optional vector of paths to          |
|                               | dependency files to include in the   |
|                               | model, as in ``r-dependencies.txt``  |
|                               | or ``conda.yaml`` .                  |
+-------------------------------+--------------------------------------+
| ``conda_env``                 | Path to Conda dependencies file.     |
+-------------------------------+--------------------------------------+

Search Runs
===========

Search for runs that satisfy expressions. Search expressions can use
Metric and Param keys.

.. code:: r

   mlflow_search_runs(filter = NULL, run_view_type = c("ACTIVE_ONLY",
     "DELETED_ONLY", "ALL"), experiment_ids = NULL, client = NULL)

.. _arguments-32:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``filter``                    | A filter expression over params,     |
|                               | metrics, and tags, allowing          |
|                               | returning a subset of runs. The      |
|                               | syntax is a subset of SQL which      |
|                               | allows only ANDing together binary   |
|                               | operations between a                 |
|                               | param/metric/tag and a constant.     |
+-------------------------------+--------------------------------------+
| ``run_view_type``             | Run view type.                       |
+-------------------------------+--------------------------------------+
| ``experiment_ids``            | List of experiment IDs to search     |
|                               | over. Attempts to use active         |
|                               | experiment if not specified.         |
+-------------------------------+--------------------------------------+
| ``client``                    | (Optional) An ``mlflow_client``      |
|                               | object.                              |
+-------------------------------+--------------------------------------+

.. _details-22:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Run MLflow Tracking Server
==========================

Wrapper for ``mlflow server``.

.. code:: r

   mlflow_server(file_store = "mlruns", default_artifact_root = NULL,
     host = "127.0.0.1", port = 5000, workers = 4,
     static_prefix = NULL)

.. _arguments-33:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``file_store``                | The root of the backing file store   |
|                               | for experiment and run data.         |
+-------------------------------+--------------------------------------+
| ``default_artifact_root``     | Local or S3 URI to store artifacts   |
|                               | in, for newly created experiments.   |
+-------------------------------+--------------------------------------+
| ``host``                      | The network address to listen on     |
|                               | (default: 127.0.0.1).                |
+-------------------------------+--------------------------------------+
| ``port``                      | The port to listen on (default:      |
|                               | 5000).                               |
+-------------------------------+--------------------------------------+
| ``workers``                   | Number of gunicorn worker processes  |
|                               | to handle requests (default: 4).     |
+-------------------------------+--------------------------------------+
| ``static_prefix``             | A prefix which will be prepended to  |
|                               | the path of all static paths.        |
+-------------------------------+--------------------------------------+

Set Experiment
==============

Sets an experiment as the active experiment. Either the name or ID of
the experiment can be provided. If the a name is provided but the
experiment does not exist, this function creates an experiment with
provided name. Returns the ID of the active experiment.

.. code:: r

   mlflow_set_experiment(experiment_name = NULL, experiment_id = NULL,
     artifact_location = NULL)

.. _arguments-34:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``experiment_name``           | Name of experiment to be activated.  |
+-------------------------------+--------------------------------------+
| ``experiment_id``             | ID of experiment to be activated.    |
+-------------------------------+--------------------------------------+
| ``artifact_location``         | Location where all artifacts for     |
|                               | this experiment are stored. If not   |
|                               | provided, the remote server will     |
|                               | select an appropriate default.       |
+-------------------------------+--------------------------------------+

Set Tag
=======

Sets a tag on a run. Tags are run metadata that can be updated during a
run and after a run completes.

.. code:: r

   mlflow_set_tag(key, value, run_id = NULL, client = NULL)

.. _arguments-35:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``key``                       | Name of the tag. Maximum size is 255 |
|                               | bytes. This field is required.       |
+-------------------------------+--------------------------------------+
| ``value``                     | String value of the tag being        |
|                               | logged. Maximum size is 500 bytes.   |
|                               | This field is required.              |
+-------------------------------+--------------------------------------+
| ``run_id``                    | Run ID.                              |
+-------------------------------+--------------------------------------+
| ``client``                    | (Optional) An ``mlflow_client``      |
|                               | object.                              |
+-------------------------------+--------------------------------------+

.. _details-23:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

Set Remote Tracking URI
=======================

Specifies the URI to the remote MLflow server that will be used to track
experiments.

.. code:: r

   mlflow_set_tracking_uri(uri)

.. _arguments-36:

Arguments
---------

+----------+--------------------------------------+
| Argument | Description                          |
+==========+======================================+
| ``uri``  | The URI to the remote MLflow server. |
+----------+--------------------------------------+

Create Dependency Snapshot
==========================

Creates a snapshot of all dependencies required to run the files in the
current directory.

.. code:: r

   mlflow_snapshot()

Source a Script with MLflow Params
==================================

This function should not be used interactively. It is designed to be
called via ``Rscript`` from the terminal or through the MLflow CLI.

.. code:: r

   mlflow_source(uri)

.. _arguments-37:

Arguments
---------

+----------+----------------------------------------------------------+
| Argument | Description                                              |
+==========+==========================================================+
| ``uri``  | Path to an R script, can be a quoted or unquoted string. |
+----------+----------------------------------------------------------+

Start Run
=========

Starts a new run. If ``client`` is not provided, this function infers
contextual information such as source name and version, and also
registers the created run as the active run. If ``client`` is provided,
no inference is done, and additional arguments such as ``user_id`` and
``start_time`` can be provided.

.. code:: r

   mlflow_start_run(run_id = NULL, experiment_id = NULL, user_id = NULL,
     start_time = NULL, tags = NULL, client = NULL)

.. _arguments-38:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``run_id``                    | If specified, get the run with the   |
|                               | specified UUID and log metrics and   |
|                               | params under that run. The run’s end |
|                               | time is unset and its status is set  |
|                               | to running, but the run’s other      |
|                               | attributes remain unchanged.         |
+-------------------------------+--------------------------------------+
| ``experiment_id``             | Used only when ``run_id`` is         |
|                               | unspecified. ID of the experiment    |
|                               | under which to create the current    |
|                               | run. If unspecified, the run is      |
|                               | created under a new experiment with  |
|                               | a randomly generated name.           |
+-------------------------------+--------------------------------------+
| ``user_id``                   | User ID or LDAP for the user         |
|                               | executing the run. Only used when    |
|                               | ``client`` is specified.             |
+-------------------------------+--------------------------------------+
| ``start_time``                | Unix timestamp of when the run       |
|                               | started in milliseconds. Only used   |
|                               | when ``client`` is specified.        |
+-------------------------------+--------------------------------------+
| ``tags``                      | Additional metadata for run in       |
|                               | key-value pairs. Only used when      |
|                               | ``client`` is specified.             |
+-------------------------------+--------------------------------------+
| ``client``                    | (Optional) An ``mlflow_client``      |
|                               | object.                              |
+-------------------------------+--------------------------------------+
| ``source_name``               | Name of the source file or URI of    |
|                               | the project to be associated with    |
|                               | the run. Defaults to the current     |
|                               | file if none provided.               |
+-------------------------------+--------------------------------------+
| ``source_version``            | Optional Git commit hash to          |
|                               | associate with the run.              |
+-------------------------------+--------------------------------------+
| ``entry_point_name``          | Optional name of the entry point for |
|                               | to the current run.                  |
+-------------------------------+--------------------------------------+
| ``source_type``               | Integer enum value describing the    |
|                               | type of the run (“local”, “project”, |
|                               | etc.).                               |
+-------------------------------+--------------------------------------+

.. _details-24:

Details
-------

When ``client`` is not specified, these functions attempt to infer the
current active client.

.. _examples-3:

Examples
--------

.. code:: r

    list("\n", "with(mlflow_start_run(), {\n", "  mlflow_log(\"test\", 10)\n", "})\n") 
    

Run MLflow User Interface
=========================

Launches the MLflow user interface.

.. code:: r

   mlflow_ui(x, ...)

.. _arguments-39:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``x``                         | An ``mlflow_client`` object.         |
+-------------------------------+--------------------------------------+
| ``...``                       | Optional arguments passed to         |
|                               | ``mlflow_server()`` when ``x`` is a  |
|                               | path to a file store.                |
+-------------------------------+--------------------------------------+

.. _examples-4:

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "mlflow_install()\n", "\n", "# launch mlflow ui locally\n", "mlflow_ui()\n", "\n", "# launch mlflow ui for existing mlflow server\n", "mlflow_set_tracking_uri(\"http://tracking-server:5000\")\n", "mlflow_ui()\n") 
    

Uninstall MLflow
================

Uninstalls MLflow by removing the Conda environment.

.. code:: r

   mlflow_uninstall()

.. _examples-5:

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "mlflow_install()\n", "mlflow_uninstall()\n") 
    
