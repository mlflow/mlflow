.. _R-api:

========
R API
========

The MLflow R API allows you to use MLflow :doc:`Tracking <tracking/>`, :doc:`Projects <projects/>` and :doc:`Models <models/>`.

For instance, you can use the R API to `install MLflow`_, start the `user interface <MLflow user interface_>`_, `create <Create Experiment_>`_ and `list experiments`_, `save models <Save Model for MLflow_>`_, `run projects <Run in MLflow_>`_ and `serve models <Serve an RFunc MLflow Model_>`_ among many other functions available in the R API.

.. contents:: Table of Contents
    :local:
    :depth: 1

Active Experiment
=================

Retrieve or set the active experiment.

.. code:: r

   mlflow_active_experiment()
   mlflow_set_active_experiment(experiment_id)

Arguments
---------

+-------------------+---------------------------------+
| Argument          | Description                     |
+===================+=================================+
| ``experiment_id`` | Identifer to get an experiment. |
+-------------------+---------------------------------+

Active Run
==========

Retrieves or sets the active run.

.. code:: r

   mlflow_active_run()
   mlflow_set_active_run(run)

.. _arguments-1:

Arguments
---------

+----------+--------------------------------+
| Argument | Description                    |
+==========+================================+
| ``run``  | The run object to make active. |
+----------+--------------------------------+

MLflow Command
==============

Executes a generic MLflow command through the commmand line interface.

.. code:: r

   mlflow_cli(..., background = FALSE, echo = TRUE)

.. _arguments-2:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``...``                       | The parameters to pass to the        |
|                               | command line.                        |
+-------------------------------+--------------------------------------+
| ``background``                | Should this command be triggered as  |
|                               | a background task? Defaults to       |
|                               | ``FALSE`` .                          |
+-------------------------------+--------------------------------------+
| ``echo``                      | Print the standard output and error  |
|                               | to the screen? Defaults to ``TRUE``  |
|                               | , does not apply to background       |
|                               | tasks.                               |
+-------------------------------+--------------------------------------+

Value
-----

A ``processx`` task.

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "mlflow_install()\n", "\n", "mlflow_cli(\"server\", \"--help\")\n") 
    

Connect to MLflow
=================

Connect to local or remote MLflow instance.

.. code:: r

   mlflow_connect(x = NULL, activate = TRUE, ...)

.. _arguments-3:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``x``                         | (Optional) Either a URL to the       |
|                               | remote MLflow server or the file     |
|                               | store, i.e. the root of the backing  |
|                               | file store for experiment and run    |
|                               | data. If not specified, will launch  |
|                               | and connect to a local instance      |
|                               | listening on a random port.          |
+-------------------------------+--------------------------------------+
| ``activate``                  | Whether to set the connction as the  |
|                               | active connection, defaults to       |
|                               | ``TRUE``.                            |
+-------------------------------+--------------------------------------+
| ``...``                       | Optional arguments passed to         |
|                               | ``mlflow_server()``.                 |
+-------------------------------+--------------------------------------+

Create Experiment
=================

Creates an MLflow experiment.

.. code:: r

   mlflow_create_experiment(name, artifact_location = NULL,
     activate = TRUE)

.. _arguments-4:

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
| ``activate``                  | Whether to set the created           |
|                               | experiment as the active experiment. |
|                               | Defaults to ``TRUE``.                |
+-------------------------------+--------------------------------------+

.. _examples-1:

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "mlflow_install()\n", "\n", "# create local experiment\n", "mlflow_create_experiment(\"My Experiment\")\n", "\n", "# create experiment in remote MLflow server\n", "mlflow_set_tracking_uri(\"http://tracking-server:5000\")\n", "mlflow_create_experiment(\"My Experiment\")\n") 
    

Create Run
==========

reate a new run within an experiment. A run is usually a single
execution of a machine learning or data ETL pipeline.

.. code:: r

   mlflow_create_run(user_id = mlflow_user(), run_name = NULL,
     source_type = NULL, source_name = NULL, status = NULL,
     start_time = NULL, end_time = NULL, source_version = NULL,
     entry_point_name = NULL, tags = NULL, experiment_id = NULL)

.. _arguments-5:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``user_id``                   | User ID or LDAP for the user         |
|                               | executing the run.                   |
+-------------------------------+--------------------------------------+
| ``run_name``                  | Human readable name for run.         |
+-------------------------------+--------------------------------------+
| ``source_type``               | Originating source for this run. One |
|                               | of Notebook, Job, Project, Local or  |
|                               | Unknown.                             |
+-------------------------------+--------------------------------------+
| ``source_name``               | String descriptor for source. For    |
|                               | example, name or description of the  |
|                               | notebook, or job name.               |
+-------------------------------+--------------------------------------+
| ``status``                    | Current status of the run. One of    |
|                               | RUNNING, SCHEDULE, FINISHED, FAILED, |
|                               | KILLED.                              |
+-------------------------------+--------------------------------------+
| ``start_time``                | Unix timestamp of when the run       |
|                               | started in milliseconds.             |
+-------------------------------+--------------------------------------+
| ``end_time``                  | Unix timestamp of when the run ended |
|                               | in milliseconds.                     |
+-------------------------------+--------------------------------------+
| ``source_version``            | Git version of the source code used  |
|                               | to create run.                       |
+-------------------------------+--------------------------------------+
| ``entry_point_name``          | Name of the entry point for the run. |
+-------------------------------+--------------------------------------+
| ``tags``                      | Additional metadata for run in       |
|                               | key-value pairs.                     |
+-------------------------------+--------------------------------------+
| ``experiment_id``             | Unique identifier for the associated |
|                               | experiment.                          |
+-------------------------------+--------------------------------------+

Details
-------

MLflow uses runs to track Param, Metric, and RunTag, associated with a
single execution.

Disconnect from MLflow
======================

Disconnects from a local MLflow instance.

.. code:: r

   mlflow_disconnect()

End Run
=======

End the active run.

.. code:: r

   mlflow_end_run(status = "FINISHED")

.. _arguments-6:

Arguments
---------

+------------+-----------------------------------------------------+
| Argument   | Description                                         |
+============+=====================================================+
| ``status`` | Ending status of the run, defaults to ``FINISHED``. |
+------------+-----------------------------------------------------+

Get Experiment
==============

Get meta data for experiment and a list of runs for this experiment.

.. code:: r

   mlflow_get_experiment(experiment_id)

.. _arguments-7:

Arguments
---------

+-------------------+---------------------------------+
| Argument          | Description                     |
+===================+=================================+
| ``experiment_id`` | Identifer to get an experiment. |
+-------------------+---------------------------------+

Get Metric History
==================

For cases that a metric is logged more than once during a run, this API
can be used to retrieve all logged values for this metric.

.. code:: r

   mlflow_get_metric_history(metric_key, run_uuid = NULL)

.. _arguments-8:

Arguments
---------

+----------------+-----------------------------------------------------+
| Argument       | Description                                         |
+================+=====================================================+
| ``metric_key`` | Name of the metric.                                 |
+----------------+-----------------------------------------------------+
| ``run_uuid``   | Unique ID for the run for which metric is recorded. |
+----------------+-----------------------------------------------------+

Get Metric
==========

API to retrieve the logged value for a metric during a run. For a run,
if this metric is logged more than once, this API will retrieve only the
latest value logged.

.. code:: r

   mlflow_get_metric(metric_key, run_uuid = NULL)

.. _arguments-9:

Arguments
---------

+----------------+-----------------------------------------------------+
| Argument       | Description                                         |
+================+=====================================================+
| ``metric_key`` | Name of the metric.                                 |
+----------------+-----------------------------------------------------+
| ``run_uuid``   | Unique ID for the run for which metric is recorded. |
+----------------+-----------------------------------------------------+

Get Param
=========

Get a param value.

.. code:: r

   mlflow_get_param(param_name, run_uuid = NULL)

.. _arguments-10:

Arguments
---------

+----------------+-------------------------------------------------------+
| Argument       | Description                                           |
+================+=======================================================+
| ``param_name`` | Name of the param. This field is required.            |
+----------------+-------------------------------------------------------+
| ``run_uuid``   | ID of the run from which to retrieve the param value. |
+----------------+-------------------------------------------------------+

.. _value-1:

Value
-----

The param value as a named list.

Get Run
=======

Get meta data, params, tags, and metrics for run. Only last logged value
for each metric is returned.

.. code:: r

   mlflow_get_run(run_uuid)

.. _arguments-11:

Arguments
---------

+--------------+------------------------+
| Argument     | Description            |
+==============+========================+
| ``run_uuid`` | Unique ID for the run. |
+--------------+------------------------+

Install MLflow
==============

Installs MLflow for individual use.

.. code:: r

   mlflow_install()

.. _details-1:

Details
-------

Notice that MLflow requires Python and Conda to be installed, see
https://www.python.org/getit/ and
https://conda.io/docs/installation.html .

.. _examples-2:

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "mlflow_install()\n") 
    

List Experiments
================

Retrieves MLflow experiments as a data frame.

.. code:: r

   mlflow_list_experiments()

.. _examples-3:

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "mlflow_install()\n", "\n", "# list local experiments\n", "mlflow_list_experiments()\n", "\n", "# list experiments in remote MLflow server\n", "mlflow_set_tracking_uri(\"http://tracking-server:5000\")\n", "mlflow_list_experiments()\n") 
    

Load MLflow Model Flavor
========================

Loads an MLflow model flavor, to be used by package authors to extend
the supported MLflow models.

.. code:: r

   mlflow_load_flavor(flavor_path)

.. _arguments-12:

Arguments
---------

+-----------------------------------+-----------------------------------+
| Argument                          | Description                       |
+===================================+===================================+
| ``flavor_path``                   | The path to the MLflow model      |
|                                   | wrapped in the correct class.     |
+-----------------------------------+-----------------------------------+

Log Artifact
============

Logs an specific file or directory as an artifact.

.. code:: r

   mlflow_log_artifact(path, artifact_path = NULL, run_uuid = NULL)

.. _arguments-13:

Arguments
---------

+-------------------+-------------------------------------------------+
| Argument          | Description                                     |
+===================+=================================================+
| ``path``          | The file or directory to log as an artifact.    |
+-------------------+-------------------------------------------------+
| ``artifact_path`` | Destination path within the run’s artifact URI. |
+-------------------+-------------------------------------------------+
| ``run_uuid``      | The run associated with this artifact.          |
+-------------------+-------------------------------------------------+

.. _details-2:

Details
-------

When logging to Amazon S3, ensure that the user has a proper policy
attach to it, for instance:

\`\`

Additionally, at least the ``AWS_ACCESS_KEY_ID`` and
``AWS_SECRET_ACCESS_KEY`` environment variables must be set to the
corresponding key and secrets provided by Amazon IAM.

Log Metric
==========

API to log a metric for a run. Metrics key-value pair that record a
single float measure. During a single execution of a run, a particular
metric can be logged several times. Backend will keep track of
historical values along with timestamps.

.. code:: r

   mlflow_log_metric(key, value, timestamp = NULL, run_uuid = NULL)

.. _arguments-14:

Arguments
---------

+-----------------------------------+-----------------------------------+
| Argument                          | Description                       |
+===================================+===================================+
| ``key``                           | Name of the metric.               |
+-----------------------------------+-----------------------------------+
| ``value``                         | Float value for the metric being  |
|                                   | logged.                           |
+-----------------------------------+-----------------------------------+
| ``timestamp``                     | Unix timestamp in milliseconds at |
|                                   | the time metric was logged.       |
+-----------------------------------+-----------------------------------+
| ``run_uuid``                      | Unique ID for the run.            |
+-----------------------------------+-----------------------------------+

Log Model
=========

Logs a model in the given run. Similar to ``mlflow_save_model()`` but
stores model as an artifact within the active run.

.. code:: r

   mlflow_log_model(fn, artifact_path, run_uuid = NULL)

.. _arguments-15:

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
| ``run_uuid``                  | The run associated with the model to |
|                               | be logged.                           |
+-------------------------------+--------------------------------------+

Log Parameter
=============

API to log a parameter used for this run. Examples are params and
hyperparams used for ML training, or constant dates and values used in
an ETL pipeline. A params is a STRING key-value pair. For a run, a
single parameter is allowed to be logged only once.

.. code:: r

   mlflow_log_param(key, value, run_uuid = NULL)

.. _arguments-16:

Arguments
---------

+--------------+--------------------------------------------------------+
| Argument     | Description                                            |
+==============+========================================================+
| ``key``      | Name of the parameter.                                 |
+--------------+--------------------------------------------------------+
| ``value``    | String value of the parameter.                         |
+--------------+--------------------------------------------------------+
| ``run_uuid`` | Unique ID for the run for which parameter is recorded. |
+--------------+--------------------------------------------------------+

Read Command Line Parameter
===========================

Reads a command line parameter.

.. code:: r

   mlflow_param(name, default = NULL, type = NULL, description = NULL)

.. _arguments-17:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``name``                      | The name for this parameter.         |
+-------------------------------+--------------------------------------+
| ``default``                   | The default value for this           |
|                               | parameter.                           |
+-------------------------------+--------------------------------------+
| ``type``                      | Type of this parameter. Required if  |
|                               | ``default`` is not set. If           |
|                               | specified, must be one of “numeric”, |
|                               | “integer”, or “string”.              |
+-------------------------------+--------------------------------------+
| ``description``               | Optional description for this        |
|                               | parameter.                           |
+-------------------------------+--------------------------------------+

Predict over MLflow Model Flavor
================================

Performs prediction over a model loaded using ``mlflow_load_model()`` ,
to be used by package authors to extend the supported MLflow models.

.. code:: r

   mlflow_predict_flavor(model, data)

.. _arguments-18:

Arguments
---------

+-----------+----------------------------------+
| Argument  | Description                      |
+===========+==================================+
| ``model`` | The loaded MLflow model flavor.  |
+-----------+----------------------------------+
| ``data``  | A data frame to perform scoring. |
+-----------+----------------------------------+

Restore Snapshot
================

Restores a snapshot of all dependencies required to run the files in the
current directory

.. code:: r

   mlflow_restore_snapshot()

Predict using RFunc MLflow Model
================================

Predict using an RFunc MLflow Model from a file or data frame.

.. code:: r

   mlflow_rfunc_predict(model_path, run_uuid = NULL, input_path = NULL,
     output_path = NULL, data = NULL, restore = FALSE)

.. _arguments-19:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``model_path``                | The path to the MLflow model, as a   |
|                               | string.                              |
+-------------------------------+--------------------------------------+
| ``run_uuid``                  | Run ID of run to grab the model      |
|                               | from.                                |
+-------------------------------+--------------------------------------+
| ``input_path``                | Path to ‘JSON’ or ‘CSV’ file to be   |
|                               | used for prediction.                 |
+-------------------------------+--------------------------------------+
| ``output_path``               | ‘JSON’ or ‘CSV’ file where the       |
|                               | prediction will be written to.       |
+-------------------------------+--------------------------------------+
| ``data``                      | Data frame to be scored. This can be |
|                               | utilized for testing purposes and    |
|                               | can only be specified when           |
|                               | ``input_path`` is not specified.     |
+-------------------------------+--------------------------------------+
| ``restore``                   | Should ``mlflow_restore_snapshot()`` |
|                               | be called before serving?            |
+-------------------------------+--------------------------------------+

.. _examples-4:

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "\n", "# save simple model which roundtrips data as prediction\n", "mlflow_save_model(function(df) df, \"mlflow_roundtrip\")\n", "\n", "# save data as json\n", "jsonlite::write_json(iris, \"iris.json\")\n", "\n", "# predict existing model from json data\n", "mlflow_rfunc_predict(\"mlflow_roundtrip\", \"iris.json\")\n") 
    

Serve an RFunc MLflow Model
===========================

Serve an RFunc MLflow Model as a local web api under
http://localhost:8090 .

.. code:: r

   mlflow_rfunc_serve(model_path, run_uuid = NULL, host = "127.0.0.1",
     port = 8090, daemonized = FALSE, browse = !daemonized,
     restore = FALSE)

.. _arguments-20:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``model_path``                | The path to the MLflow model, as a   |
|                               | string.                              |
+-------------------------------+--------------------------------------+
| ``run_uuid``                  | ID of run to grab the model from.    |
+-------------------------------+--------------------------------------+
| ``host``                      | Address to use to serve model, as a  |
|                               | string.                              |
+-------------------------------+--------------------------------------+
| ``port``                      | Port to use to serve model, as       |
|                               | numeric.                             |
+-------------------------------+--------------------------------------+
| ``daemonized``                | Makes ‘httpuv’ server daemonized so  |
|                               | R interactive sessions are not       |
|                               | blocked to handle requests. To       |
|                               | terminate a daemonized server, call  |
|                               | ‘httpuv::stopDaemonizedServer()’     |
|                               | with the handle returned from this   |
|                               | call.                                |
+-------------------------------+--------------------------------------+
| ``browse``                    | Launch browser with serving landing  |
|                               | page?                                |
+-------------------------------+--------------------------------------+
| ``restore``                   | Should ``mlflow_restore_snapshot()`` |
|                               | be called before serving?            |
+-------------------------------+--------------------------------------+

.. _examples-5:

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "\n", "# save simple model with constant prediction\n", "mlflow_save_model(function(df) 1, \"mlflow_constant\")\n", "\n", "# serve an existing model over a web interface\n", "mlflow_rfunc_serve(\"mlflow_constant\")\n", "\n", "# request prediction from server\n", "httr::POST(\"http://127.0.0.1:8090/predict/\")\n") 

Run in MLflow
=============

Wrapper for ``mlflow run``.

.. code:: r

   mlflow_run(uri = ".", entry_point = NULL, version = NULL,
     param_list = NULL, experiment_id = NULL, mode = NULL,
     cluster_spec = NULL, git_username = NULL, git_password = NULL,
     no_conda = FALSE, storage_dir = NULL)

.. _arguments-21:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``uri``                       | A directory containing modeling      |
|                               | scripts, defaults to the current     |
|                               | directory.                           |
+-------------------------------+--------------------------------------+
| ``entry_point``               | Entry point within project, defaults |
|                               | to ``main`` if not specified.        |
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
| ``mode``                      | Execution mode to use for run.       |
+-------------------------------+--------------------------------------+
| ``cluster_spec``              | Path to JSON file describing the     |
|                               | cluster to use when launching a run  |
|                               | on Databricks.                       |
+-------------------------------+--------------------------------------+
| ``git_username``              | Username for HTTP(S) Git             |
|                               | authentication.                      |
+-------------------------------+--------------------------------------+
| ``git_password``              | Password for HTTP(S) Git             |
|                               | authentication.                      |
+-------------------------------+--------------------------------------+
| ``no_conda``                  | If specified, assume that MLflow is  |
|                               | running within a Conda environment   |
|                               | with the necessary dependencies for  |
|                               | the current project instead of       |
|                               | attempting to create a new conda     |
|                               | environment. Only valid if running   |
|                               | locally.                             |
+-------------------------------+--------------------------------------+
| ``storage_dir``               | Only valid when ``mode`` is local.   |
|                               | MLflow downloads artifacts from      |
|                               | distributed URIs passed to           |
|                               | parameters of type ‘path’ to         |
|                               | subdirectories of storage_dir.       |
+-------------------------------+--------------------------------------+

.. _value-2:

Value
-----

The run associated with this run.

Save MLflow Model Flavor
========================

Saves model in MLflow’s flavor, to be used by package authors to extend
the supported MLflow models.

.. code:: r

   mlflow_save_flavor(x, path = "model")

.. _arguments-22:

Arguments
---------

+-----------------------------------+-----------------------------------+
| Argument                          | Description                       |
+===================================+===================================+
| ``x``                             | The serving function or model     |
|                                   | that will perform a prediction.   |
+-----------------------------------+-----------------------------------+
| ``path``                          | Destination path where this       |
|                                   | MLflow compatible model will be   |
|                                   | saved.                            |
+-----------------------------------+-----------------------------------+

.. _value-3:

Value
-----

This funciton must return a list of flavors that conform to the MLmodel
specification.

Save Model for MLflow
=====================

Saves model in MLflow’s format that can later be used for prediction and
serving.

.. code:: r

   mlflow_save_model(x, path = "model", dependencies = NULL)

.. _arguments-23:

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
| ``dependencies``              | Optional vector of paths to          |
|                               | dependency files to include in the   |
|                               | model, as in ``r-dependencies.txt``  |
|                               | or ``conda.yaml`` .                  |
+-------------------------------+--------------------------------------+

Run the MLflow Tracking Server
==============================

Wrapper for ``mlflow server``.

.. code:: r

   mlflow_server(file_store = "mlruns", default_artifact_root = NULL,
     host = "127.0.0.1", port = 5000, workers = 4,
     static_prefix = NULL)

.. _arguments-24:

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

Set Remote Tracking URI
=======================

Specifies the URI to the remote MLflow server that will be used to track
experiments.

.. code:: r

   mlflow_set_tracking_uri(uri)

.. _arguments-25:

Arguments
---------

+----------+--------------------------------------+
| Argument | Description                          |
+==========+======================================+
| ``uri``  | The URI to the remote MLflow server. |
+----------+--------------------------------------+

Dependencies Snapshot
=====================

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

.. _arguments-26:

Arguments
---------

+----------+----------------------------------------------------------+
| Argument | Description                                              |
+==========+==========================================================+
| ``uri``  | Path to an R script, can be a quoted or unquoted string. |
+----------+----------------------------------------------------------+

Start Run
=========

Starts a new run within an experiment, should be used within a ``with``
block.

.. code:: r

   mlflow_start_run(run_uuid = NULL, experiment_id = NULL,
     source_name = NULL, source_version = NULL, entry_point_name = NULL,
     source_type = "LOCAL")

.. _arguments-27:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``run_uuid``                  | If specified, get the run with the   |
|                               | specified UUID and log metrics and   |
|                               | params under that run. The run’s end |
|                               | time is unset and its status is set  |
|                               | to running, but the run’s other      |
|                               | attributes remain unchanged.         |
+-------------------------------+--------------------------------------+
| ``experiment_id``             | Used only when ``run_uuid`` is       |
|                               | unspecified. ID of the experiment    |
|                               | under which to create the current    |
|                               | run. If unspecified, the run is      |
|                               | created under a new experiment with  |
|                               | a randomly generated name.           |
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

.. _examples-6:

Examples
--------

.. code:: r

    list("\n", "with(mlflow_start_run(), {\n", "  mlflow_log(\"test\", 10)\n", "})\n") 
    

Get Remote Tracking URI
=======================

Get Remote Tracking URI

.. code:: r

   mlflow_tracking_uri()

MLflow User Interface
=====================

Launches MLflow user interface.

.. code:: r

   mlflow_ui(x, ...)

.. _arguments-28:

Arguments
---------

+-------------------------------+--------------------------------------+
| Argument                      | Description                          |
+===============================+======================================+
| ``x``                         | If specified, can be either an       |
|                               | ``mlflow_connection`` object or a    |
|                               | string specifying the file store,    |
|                               | i.e. the root of the backing file    |
|                               | store for experiment and run data.   |
+-------------------------------+--------------------------------------+
| ``...``                       | Optional arguments passed to         |
|                               | ``mlflow_server()`` when ``x`` is a  |
|                               | path to a file store.                |
+-------------------------------+--------------------------------------+

.. _examples-7:

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "mlflow_install()\n", "\n", "# launch mlflow ui locally\n", "mlflow_ui()\n", "\n", "# launch mlflow ui for existing mlflow server\n", "mlflow_set_tracking_uri(\"http://tracking-server:5000\")\n", "mlflow_ui()\n") 
    

Uninstalls MLflow.
==================

Uninstalls MLflow by removing the Conda environment.

.. code:: r

   mlflow_uninstall()

.. _examples-8:

Examples
--------

.. code:: r

    list("\n", "library(mlflow)\n", "mlflow_install()\n", "mlflow_uninstall()\n") 
    

Update Run
==========

Update Run

.. code:: r

   mlflow_update_run(status = c("FINISHED", "SCHEDULED", "FAILED",
     "KILLED"), end_time = NULL, run_uuid = NULL)

.. _arguments-29:

Arguments
---------

+--------------+-------------------------------------------------------+
| Argument     | Description                                           |
+==============+=======================================================+
| ``status``   | Updated status of the run. Defaults to ``FINISHED``.  |
+--------------+-------------------------------------------------------+
| ``end_time`` | Unix timestamp of when the run ended in milliseconds. |
+--------------+-------------------------------------------------------+
| ``run_uuid`` | Unique identifier for the run.                        |
+--------------+-------------------------------------------------------+
