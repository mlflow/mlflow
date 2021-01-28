
.. _rest-api:

========
REST API
========


The MLflow REST API allows you to create, list, and get experiments and runs, and log parameters, metrics, and artifacts.
The API is hosted under the ``/api`` route on the MLflow tracking server. For example, to list
experiments on a tracking server hosted at ``http://localhost:5000``, access
``http://localhost:5000/api/2.0/preview/mlflow/experiments/list``.

MLflow also provides a health check endpoint at the ``/health`` route, which responds with a 200 response code and
``OK`` in the response body.

.. contents:: Table of Contents
    :local:
    :depth: 1

===========================



.. _mlflowMlflowServicecreateExperiment:

Create Experiment
=================


+-----------------------------------+-------------+
|             Endpoint              | HTTP Method |
+===================================+=============+
| ``2.0/mlflow/experiments/create`` | ``POST``    |
+-----------------------------------+-------------+

Create an experiment with a name. Returns the ID of the newly created experiment.
Validates that another experiment with the same name does not already exist and fails
if another experiment with the same name already exists.


Throws ``RESOURCE_ALREADY_EXISTS`` if a experiment with the given name exists.




.. _mlflowCreateExperiment:

Request Structure
-----------------






+-------------------+------------+------------------------------------------------------------------------+
|    Field Name     |    Type    |                              Description                               |
+===================+============+========================================================================+
| name              | ``STRING`` | Experiment name.                                                       |
|                   |            | This field is required.                                                |
|                   |            |                                                                        |
+-------------------+------------+------------------------------------------------------------------------+
| artifact_location | ``STRING`` | Location where all artifacts for the experiment are stored.            |
|                   |            | If not provided, the remote server will select an appropriate default. |
+-------------------+------------+------------------------------------------------------------------------+

.. _mlflowCreateExperimentResponse:

Response Structure
------------------






+---------------+------------+---------------------------------------+
|  Field Name   |    Type    |              Description              |
+===============+============+=======================================+
| experiment_id | ``STRING`` | Unique identifier for the experiment. |
+---------------+------------+---------------------------------------+

===========================



.. _mlflowMlflowServicelistExperiments:

List Experiments
================


+---------------------------------+-------------+
|            Endpoint             | HTTP Method |
+=================================+=============+
| ``2.0/mlflow/experiments/list`` | ``GET``     |
+---------------------------------+-------------+

Get a list of all experiments.




.. _mlflowListExperiments:

Request Structure
-----------------






+------------+-----------------------+---------------------------------------------------+
| Field Name |         Type          |                    Description                    |
+============+=======================+===================================================+
| view_type  | :ref:`mlflowviewtype` | Qualifier for type of experiments to be returned. |
|            |                       | If unspecified, return only active experiments.   |
+------------+-----------------------+---------------------------------------------------+

.. _mlflowListExperimentsResponse:

Response Structure
------------------






+-------------+-------------------------------------+------------------+
| Field Name  |                Type                 |   Description    |
+=============+=====================================+==================+
| experiments | An array of :ref:`mlflowexperiment` | All experiments. |
+-------------+-------------------------------------+------------------+

===========================



.. _mlflowMlflowServicegetExperiment:

Get Experiment
==============


+--------------------------------+-------------+
|            Endpoint            | HTTP Method |
+================================+=============+
| ``2.0/mlflow/experiments/get`` | ``GET``     |
+--------------------------------+-------------+

Get metadata for an experiment. This method works on deleted experiments.




.. _mlflowGetExperiment:

Request Structure
-----------------






+---------------+------------+----------------------------------+
|  Field Name   |    Type    |           Description            |
+===============+============+==================================+
| experiment_id | ``STRING`` | ID of the associated experiment. |
|               |            | This field is required.          |
|               |            |                                  |
+---------------+------------+----------------------------------+

.. _mlflowGetExperimentResponse:

Response Structure
------------------






+------------+----------------------------------+---------------------------------------------------------------------------+
| Field Name |               Type               |                                Description                                |
+============+==================================+===========================================================================+
| experiment | :ref:`mlflowexperiment`          | Experiment details.                                                       |
+------------+----------------------------------+---------------------------------------------------------------------------+
| runs       | An array of :ref:`mlflowruninfo` | A collection of active runs in the experiment. Note: this may not contain |
|            |                                  | all of the experiment's active runs.                                      |
|            |                                  |                                                                           |
|            |                                  | This field is deprecated. Please use the "Search Runs" API to fetch       |
|            |                                  | runs within an experiment.                                                |
+------------+----------------------------------+---------------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicegetExperimentByName:

Get Experiment By Name
======================


+----------------------------------------+-------------+
|                Endpoint                | HTTP Method |
+========================================+=============+
| ``2.0/mlflow/experiments/get-by-name`` | ``GET``     |
+----------------------------------------+-------------+

Get metadata for an experiment.

This endpoint will return deleted experiments, but prefers the active experiment
if an active and deleted experiment share the same name. If multiple deleted
experiments share the same name, the API will return one of them.

Throws ``RESOURCE_DOES_NOT_EXIST`` if no experiment with the specified name exists.




.. _mlflowGetExperimentByName:

Request Structure
-----------------






+-----------------+------------+------------------------------------+
|   Field Name    |    Type    |            Description             |
+=================+============+====================================+
| experiment_name | ``STRING`` | Name of the associated experiment. |
|                 |            | This field is required.            |
|                 |            |                                    |
+-----------------+------------+------------------------------------+

.. _mlflowGetExperimentByNameResponse:

Response Structure
------------------






+------------+-------------------------+---------------------+
| Field Name |          Type           |     Description     |
+============+=========================+=====================+
| experiment | :ref:`mlflowexperiment` | Experiment details. |
+------------+-------------------------+---------------------+

===========================



.. _mlflowMlflowServicedeleteExperiment:

Delete Experiment
=================


+-----------------------------------+-------------+
|             Endpoint              | HTTP Method |
+===================================+=============+
| ``2.0/mlflow/experiments/delete`` | ``POST``    |
+-----------------------------------+-------------+

Mark an experiment and associated metadata, runs, metrics, params, and tags for deletion.
If the experiment uses FileStore, artifacts associated with experiment are also deleted.




.. _mlflowDeleteExperiment:

Request Structure
-----------------






+---------------+------------+----------------------------------+
|  Field Name   |    Type    |           Description            |
+===============+============+==================================+
| experiment_id | ``STRING`` | ID of the associated experiment. |
|               |            | This field is required.          |
|               |            |                                  |
+---------------+------------+----------------------------------+

===========================



.. _mlflowMlflowServicerestoreExperiment:

Restore Experiment
==================


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/mlflow/experiments/restore`` | ``POST``    |
+------------------------------------+-------------+

Restore an experiment marked for deletion. This also restores
associated metadata, runs, metrics, params, and tags. If experiment uses FileStore, underlying
artifacts associated with experiment are also restored.

Throws ``RESOURCE_DOES_NOT_EXIST`` if experiment was never created or was permanently deleted.




.. _mlflowRestoreExperiment:

Request Structure
-----------------






+---------------+------------+----------------------------------+
|  Field Name   |    Type    |           Description            |
+===============+============+==================================+
| experiment_id | ``STRING`` | ID of the associated experiment. |
|               |            | This field is required.          |
|               |            |                                  |
+---------------+------------+----------------------------------+

===========================



.. _mlflowMlflowServiceupdateExperiment:

Update Experiment
=================


+-----------------------------------+-------------+
|             Endpoint              | HTTP Method |
+===================================+=============+
| ``2.0/mlflow/experiments/update`` | ``POST``    |
+-----------------------------------+-------------+

Update experiment metadata.




.. _mlflowUpdateExperiment:

Request Structure
-----------------






+---------------+------------+---------------------------------------------------------------------------------------------+
|  Field Name   |    Type    |                                         Description                                         |
+===============+============+=============================================================================================+
| experiment_id | ``STRING`` | ID of the associated experiment.                                                            |
|               |            | This field is required.                                                                     |
|               |            |                                                                                             |
+---------------+------------+---------------------------------------------------------------------------------------------+
| new_name      | ``STRING`` | If provided, the experiment's name is changed to the new name. The new name must be unique. |
+---------------+------------+---------------------------------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicecreateRun:

Create Run
==========


+----------------------------+-------------+
|          Endpoint          | HTTP Method |
+============================+=============+
| ``2.0/mlflow/runs/create`` | ``POST``    |
+----------------------------+-------------+

Create a new run within an experiment. A run is usually a single execution of a
machine learning or data ETL pipeline. MLflow uses runs to track :ref:`mlflowParam`,
:ref:`mlflowMetric`, and :ref:`mlflowRunTag` associated with a single execution.




.. _mlflowCreateRun:

Request Structure
-----------------






+---------------+---------------------------------+----------------------------------------------------------------------------+
|  Field Name   |              Type               |                                Description                                 |
+===============+=================================+============================================================================+
| experiment_id | ``STRING``                      | ID of the associated experiment.                                           |
+---------------+---------------------------------+----------------------------------------------------------------------------+
| user_id       | ``STRING``                      | ID of the user executing the run.                                          |
|               |                                 | This field is deprecated as of MLflow 1.0, and will be removed in a future |
|               |                                 | MLflow release. Use 'mlflow.user' tag instead.                             |
+---------------+---------------------------------+----------------------------------------------------------------------------+
| start_time    | ``INT64``                       | Unix timestamp in milliseconds of when the run started.                    |
+---------------+---------------------------------+----------------------------------------------------------------------------+
| tags          | An array of :ref:`mlflowruntag` | Additional metadata for run.                                               |
+---------------+---------------------------------+----------------------------------------------------------------------------+

.. _mlflowCreateRunResponse:

Response Structure
------------------






+------------+------------------+------------------------+
| Field Name |       Type       |      Description       |
+============+==================+========================+
| run        | :ref:`mlflowrun` | The newly created run. |
+------------+------------------+------------------------+

===========================



.. _mlflowMlflowServicedeleteRun:

Delete Run
==========


+----------------------------+-------------+
|          Endpoint          | HTTP Method |
+============================+=============+
| ``2.0/mlflow/runs/delete`` | ``POST``    |
+----------------------------+-------------+

Mark a run for deletion.




.. _mlflowDeleteRun:

Request Structure
-----------------






+------------+------------+--------------------------+
| Field Name |    Type    |       Description        |
+============+============+==========================+
| run_id     | ``STRING`` | ID of the run to delete. |
|            |            | This field is required.  |
|            |            |                          |
+------------+------------+--------------------------+

===========================



.. _mlflowMlflowServicerestoreRun:

Restore Run
===========


+-----------------------------+-------------+
|          Endpoint           | HTTP Method |
+=============================+=============+
| ``2.0/mlflow/runs/restore`` | ``POST``    |
+-----------------------------+-------------+

Restore a deleted run.




.. _mlflowRestoreRun:

Request Structure
-----------------






+------------+------------+---------------------------+
| Field Name |    Type    |        Description        |
+============+============+===========================+
| run_id     | ``STRING`` | ID of the run to restore. |
|            |            | This field is required.   |
|            |            |                           |
+------------+------------+---------------------------+

===========================



.. _mlflowMlflowServicegetRun:

Get Run
=======


+-------------------------+-------------+
|        Endpoint         | HTTP Method |
+=========================+=============+
| ``2.0/mlflow/runs/get`` | ``GET``     |
+-------------------------+-------------+

Get metadata, metrics, params, and tags for a run. In the case where multiple metrics
with the same key are logged for a run, return only the value with the latest timestamp.
If there are multiple values with the latest timestamp, return the maximum of these values.




.. _mlflowGetRun:

Request Structure
-----------------






+------------+------------+--------------------------------------------------------------------------+
| Field Name |    Type    |                               Description                                |
+============+============+==========================================================================+
| run_id     | ``STRING`` | ID of the run to fetch. Must be provided.                                |
+------------+------------+--------------------------------------------------------------------------+
| run_uuid   | ``STRING`` | [Deprecated, use run_id instead] ID of the run to fetch. This field will |
|            |            | be removed in a future MLflow version.                                   |
+------------+------------+--------------------------------------------------------------------------+

.. _mlflowGetRunResponse:

Response Structure
------------------






+------------+------------------+----------------------------------------------------------------------------+
| Field Name |       Type       |                                Description                                 |
+============+==================+============================================================================+
| run        | :ref:`mlflowrun` | Run metadata (name, start time, etc) and data (metrics, params, and tags). |
+------------+------------------+----------------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicelogMetric:

Log Metric
==========


+--------------------------------+-------------+
|            Endpoint            | HTTP Method |
+================================+=============+
| ``2.0/mlflow/runs/log-metric`` | ``POST``    |
+--------------------------------+-------------+

Log a metric for a run. A metric is a key-value pair (string key, float value) with an
associated timestamp. Examples include the various metrics that represent ML model accuracy.
A metric can be logged multiple times.




.. _mlflowLogMetric:

Request Structure
-----------------






+------------+------------+-----------------------------------------------------------------------------------------------+
| Field Name |    Type    |                                          Description                                          |
+============+============+===============================================================================================+
| run_id     | ``STRING`` | ID of the run under which to log the metric. Must be provided.                                |
+------------+------------+-----------------------------------------------------------------------------------------------+
| run_uuid   | ``STRING`` | [Deprecated, use run_id instead] ID of the run under which to log the metric. This field will |
|            |            | be removed in a future MLflow version.                                                        |
+------------+------------+-----------------------------------------------------------------------------------------------+
| key        | ``STRING`` | Name of the metric.                                                                           |
|            |            | This field is required.                                                                       |
|            |            |                                                                                               |
+------------+------------+-----------------------------------------------------------------------------------------------+
| value      | ``DOUBLE`` | Double value of the metric being logged.                                                      |
|            |            | This field is required.                                                                       |
|            |            |                                                                                               |
+------------+------------+-----------------------------------------------------------------------------------------------+
| timestamp  | ``INT64``  | Unix timestamp in milliseconds at the time metric was logged.                                 |
|            |            | This field is required.                                                                       |
|            |            |                                                                                               |
+------------+------------+-----------------------------------------------------------------------------------------------+
| step       | ``INT64``  | Step at which to log the metric                                                               |
+------------+------------+-----------------------------------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicelogBatch:

Log Batch
=========


+-------------------------------+-------------+
|           Endpoint            | HTTP Method |
+===============================+=============+
| ``2.0/mlflow/runs/log-batch`` | ``POST``    |
+-------------------------------+-------------+

Log a batch of metrics, params, and tags for a run.
If any data failed to be persisted, the server will respond with an error (non-200 status code).
In case of error (due to internal server error or an invalid request), partial data may
be written.

You can write metrics, params, and tags in interleaving fashion, but within a given entity
type are guaranteed to follow the order specified in the request body. That is, for an API
request like

.. code-block:: json

  {
     "run_id": "2a14ed5c6a87499199e0106c3501eab8",
     "metrics": [
       {"key": "mae", "value": 2.5, "timestamp": 1552550804},
       {"key": "rmse", "value": 2.7, "timestamp": 1552550804},
     ],
     "params": [
       {"key": "model_class", "value": "LogisticRegression"},
     ]
  }

the server is guaranteed to write metric "rmse" after "mae", though it may write param
"model_class" before both metrics, after "mae", or after both metrics.

The overwrite behavior for metrics, params, and tags is as follows:

- Metrics: metric values are never overwritten. Logging a metric (key, value, timestamp) appends to the set of values for the metric with the provided key.

- Tags: tag values can be overwritten by successive writes to the same tag key. That is, if multiple tag values with the same key are provided in the same API request, the last-provided tag value is written. Logging the same tag (key, value) is permitted - that is, logging a tag is idempotent.

- Params: once written, param values cannot be changed (attempting to overwrite a param value will result in an error). However, logging the same param (key, value) is permitted - that is, logging a param is idempotent.

Request Limits
--------------
A single JSON-serialized API request may be up to 1 MB in size and contain:

- No more than 1000 metrics, params, and tags in total
- Up to 1000 metrics
- Up to 100 params
- Up to 100 tags

For example, a valid request might contain 900 metrics, 50 params, and 50 tags, but logging
900 metrics, 50 params, and 51 tags is invalid. The following limits also apply
to metric, param, and tag keys and values:

- Metric, param, and tag keys can be up to 250 characters in length
- Param and tag values can be up to 250 characters in length




.. _mlflowLogBatch:

Request Structure
-----------------






+------------+---------------------------------+---------------------------------------------------------------------------------+
| Field Name |              Type               |                                   Description                                   |
+============+=================================+=================================================================================+
| run_id     | ``STRING``                      | ID of the run to log under                                                      |
+------------+---------------------------------+---------------------------------------------------------------------------------+
| metrics    | An array of :ref:`mlflowmetric` | Metrics to log. A single request can contain up to 1000 metrics, and up to 1000 |
|            |                                 | metrics, params, and tags in total.                                             |
+------------+---------------------------------+---------------------------------------------------------------------------------+
| params     | An array of :ref:`mlflowparam`  | Params to log. A single request can contain up to 100 params, and up to 1000    |
|            |                                 | metrics, params, and tags in total.                                             |
+------------+---------------------------------+---------------------------------------------------------------------------------+
| tags       | An array of :ref:`mlflowruntag` | Tags to log. A single request can contain up to 100 tags, and up to 1000        |
|            |                                 | metrics, params, and tags in total.                                             |
+------------+---------------------------------+---------------------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicesetExperimentTag:

Set Experiment Tag
==================


+-----------------------------------------------+-------------+
|                   Endpoint                    | HTTP Method |
+===============================================+=============+
| ``2.0/mlflow/experiments/set-experiment-tag`` | ``POST``    |
+-----------------------------------------------+-------------+

Set a tag on an experiment. Experiment tags are metadata that can be updated.




.. _mlflowSetExperimentTag:

Request Structure
-----------------






+---------------+------------+-------------------------------------------------------------------------------------+
|  Field Name   |    Type    |                                     Description                                     |
+===============+============+=====================================================================================+
| experiment_id | ``STRING`` | ID of the experiment under which to log the tag. Must be provided.                  |
|               |            | This field is required.                                                             |
|               |            |                                                                                     |
+---------------+------------+-------------------------------------------------------------------------------------+
| key           | ``STRING`` | Name of the tag. Maximum size depends on storage backend.                           |
|               |            | All storage backends are guaranteed to support key values up to 250 bytes in size.  |
|               |            | This field is required.                                                             |
|               |            |                                                                                     |
+---------------+------------+-------------------------------------------------------------------------------------+
| value         | ``STRING`` | String value of the tag being logged. Maximum size depends on storage backend.      |
|               |            | All storage backends are guaranteed to support key values up to 5000 bytes in size. |
|               |            | This field is required.                                                             |
|               |            |                                                                                     |
+---------------+------------+-------------------------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicesetTag:

Set Tag
=======


+-----------------------------+-------------+
|          Endpoint           | HTTP Method |
+=============================+=============+
| ``2.0/mlflow/runs/set-tag`` | ``POST``    |
+-----------------------------+-------------+

Set a tag on a run. Tags are run metadata that can be updated during a run and after
a run completes.




.. _mlflowSetTag:

Request Structure
-----------------






+------------+------------+--------------------------------------------------------------------------------------------+
| Field Name |    Type    |                                        Description                                         |
+============+============+============================================================================================+
| run_id     | ``STRING`` | ID of the run under which to log the tag. Must be provided.                                |
+------------+------------+--------------------------------------------------------------------------------------------+
| run_uuid   | ``STRING`` | [Deprecated, use run_id instead] ID of the run under which to log the tag. This field will |
|            |            | be removed in a future MLflow version.                                                     |
+------------+------------+--------------------------------------------------------------------------------------------+
| key        | ``STRING`` | Name of the tag. Maximum size depends on storage backend.                                  |
|            |            | All storage backends are guaranteed to support key values up to 250 bytes in size.         |
|            |            | This field is required.                                                                    |
|            |            |                                                                                            |
+------------+------------+--------------------------------------------------------------------------------------------+
| value      | ``STRING`` | String value of the tag being logged. Maximum size depends on storage backend.             |
|            |            | All storage backends are guaranteed to support key values up to 5000 bytes in size.        |
|            |            | This field is required.                                                                    |
|            |            |                                                                                            |
+------------+------------+--------------------------------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicedeleteTag:

Delete Tag
==========


+--------------------------------+-------------+
|            Endpoint            | HTTP Method |
+================================+=============+
| ``2.0/mlflow/runs/delete-tag`` | ``POST``    |
+--------------------------------+-------------+

Delete a tag on a run. Tags are run metadata that can be updated during a run and after
a run completes.




.. _mlflowDeleteTag:

Request Structure
-----------------






+------------+------------+----------------------------------------------------------------+
| Field Name |    Type    |                          Description                           |
+============+============+================================================================+
| run_id     | ``STRING`` | ID of the run that the tag was logged under. Must be provided. |
|            |            | This field is required.                                        |
|            |            |                                                                |
+------------+------------+----------------------------------------------------------------+
| key        | ``STRING`` | Name of the tag. Maximum size is 255 bytes. Must be provided.  |
|            |            | This field is required.                                        |
|            |            |                                                                |
+------------+------------+----------------------------------------------------------------+

===========================



.. _mlflowMlflowServicelogParam:

Log Param
=========


+-----------------------------------+-------------+
|             Endpoint              | HTTP Method |
+===================================+=============+
| ``2.0/mlflow/runs/log-parameter`` | ``POST``    |
+-----------------------------------+-------------+

Log a param used for a run. A param is a key-value pair (string key,
string value). Examples include hyperparameters used for ML model training and
constant dates and values used in an ETL pipeline. A param can be logged only once for a run.




.. _mlflowLogParam:

Request Structure
-----------------






+------------+------------+----------------------------------------------------------------------------------------------+
| Field Name |    Type    |                                         Description                                          |
+============+============+==============================================================================================+
| run_id     | ``STRING`` | ID of the run under which to log the param. Must be provided.                                |
+------------+------------+----------------------------------------------------------------------------------------------+
| run_uuid   | ``STRING`` | [Deprecated, use run_id instead] ID of the run under which to log the param. This field will |
|            |            | be removed in a future MLflow version.                                                       |
+------------+------------+----------------------------------------------------------------------------------------------+
| key        | ``STRING`` | Name of the param. Maximum size is 255 bytes.                                                |
|            |            | This field is required.                                                                      |
|            |            |                                                                                              |
+------------+------------+----------------------------------------------------------------------------------------------+
| value      | ``STRING`` | String value of the param being logged. Maximum size is 500 bytes.                           |
|            |            | This field is required.                                                                      |
|            |            |                                                                                              |
+------------+------------+----------------------------------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicegetMetricHistory:

Get Metric History
==================


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/mlflow/metrics/get-history`` | ``GET``     |
+------------------------------------+-------------+

Get a list of all values for the specified metric for a given run.




.. _mlflowGetMetricHistory:

Request Structure
-----------------






+------------+------------+----------------------------------------------------------------------------------------------+
| Field Name |    Type    |                                         Description                                          |
+============+============+==============================================================================================+
| run_id     | ``STRING`` | ID of the run from which to fetch metric values. Must be provided.                           |
+------------+------------+----------------------------------------------------------------------------------------------+
| run_uuid   | ``STRING`` | [Deprecated, use run_id instead] ID of the run from which to fetch metric values. This field |
|            |            | will be removed in a future MLflow version.                                                  |
+------------+------------+----------------------------------------------------------------------------------------------+
| metric_key | ``STRING`` | Name of the metric.                                                                          |
|            |            | This field is required.                                                                      |
|            |            |                                                                                              |
+------------+------------+----------------------------------------------------------------------------------------------+

.. _mlflowGetMetricHistoryResponse:

Response Structure
------------------






+------------+---------------------------------+------------------------------------+
| Field Name |              Type               |            Description             |
+============+=================================+====================================+
| metrics    | An array of :ref:`mlflowmetric` | All logged values for this metric. |
+------------+---------------------------------+------------------------------------+

===========================



.. _mlflowMlflowServicesearchRuns:

Search Runs
===========


+----------------------------+-------------+
|          Endpoint          | HTTP Method |
+============================+=============+
| ``2.0/mlflow/runs/search`` | ``POST``    |
+----------------------------+-------------+

Search for runs that satisfy expressions. Search expressions can use :ref:`mlflowMetric` and
:ref:`mlflowParam` keys.




.. _mlflowSearchRuns:

Request Structure
-----------------






+----------------+------------------------+------------------------------------------------------------------------------------------------------+
|   Field Name   |          Type          |                                             Description                                              |
+================+========================+======================================================================================================+
| experiment_ids | An array of ``STRING`` | List of experiment IDs to search over.                                                               |
+----------------+------------------------+------------------------------------------------------------------------------------------------------+
| filter         | ``STRING``             | A filter expression over params, metrics, and tags, that allows returning a subset of                |
|                |                        | runs. The syntax is a subset of SQL that supports ANDing together binary operations                  |
|                |                        | between a param, metric, or tag and a constant.                                                      |
|                |                        |                                                                                                      |
|                |                        | Example: ``metrics.rmse < 1 and params.model_class = 'LogisticRegression'``                          |
|                |                        |                                                                                                      |
|                |                        | You can select columns with special characters (hyphen, space, period, etc.) by using double quotes: |
|                |                        | ``metrics."model class" = 'LinearRegression' and tags."user-name" = 'Tomas'``                        |
|                |                        |                                                                                                      |
|                |                        | Supported operators are ``=``, ``!=``, ``>``, ``>=``, ``<``, and ``<=``.                             |
+----------------+------------------------+------------------------------------------------------------------------------------------------------+
| run_view_type  | :ref:`mlflowviewtype`  | Whether to display only active, only deleted, or all runs.                                           |
|                |                        | Defaults to only active runs.                                                                        |
+----------------+------------------------+------------------------------------------------------------------------------------------------------+
| max_results    | ``INT32``              | Maximum number of runs desired. Max threshold is 50000                                               |
+----------------+------------------------+------------------------------------------------------------------------------------------------------+
| order_by       | An array of ``STRING`` | List of columns to be ordered by, including attributes, params, metrics, and tags with an            |
|                |                        | optional "DESC" or "ASC" annotation, where "ASC" is the default.                                     |
|                |                        | Example: ["params.input DESC", "metrics.alpha ASC", "metrics.rmse"]                                  |
|                |                        | Tiebreaks are done by start_time DESC followed by run_id for runs with the same start time           |
|                |                        | (and this is the default ordering criterion if order_by is not provided).                            |
+----------------+------------------------+------------------------------------------------------------------------------------------------------+
| page_token     | ``STRING``             |                                                                                                      |
+----------------+------------------------+------------------------------------------------------------------------------------------------------+

.. _mlflowSearchRunsResponse:

Response Structure
------------------






+-----------------+------------------------------+--------------------------------------+
|   Field Name    |             Type             |             Description              |
+=================+==============================+======================================+
| runs            | An array of :ref:`mlflowrun` | Runs that match the search criteria. |
+-----------------+------------------------------+--------------------------------------+
| next_page_token | ``STRING``                   |                                      |
+-----------------+------------------------------+--------------------------------------+

===========================



.. _mlflowMlflowServicelistArtifacts:

List Artifacts
==============


+-------------------------------+-------------+
|           Endpoint            | HTTP Method |
+===============================+=============+
| ``2.0/mlflow/artifacts/list`` | ``GET``     |
+-------------------------------+-------------+

List artifacts for a run. Takes an optional ``artifact_path`` prefix which if specified,
the response contains only artifacts with the specified prefix.




.. _mlflowListArtifacts:

Request Structure
-----------------






+------------+------------+-----------------------------------------------------------------------------------------+
| Field Name |    Type    |                                       Description                                       |
+============+============+=========================================================================================+
| run_id     | ``STRING`` | ID of the run whose artifacts to list. Must be provided.                                |
+------------+------------+-----------------------------------------------------------------------------------------+
| run_uuid   | ``STRING`` | [Deprecated, use run_id instead] ID of the run whose artifacts to list. This field will |
|            |            | be removed in a future MLflow version.                                                  |
+------------+------------+-----------------------------------------------------------------------------------------+
| path       | ``STRING`` | Filter artifacts matching this path (a relative path from the root artifact directory). |
+------------+------------+-----------------------------------------------------------------------------------------+

.. _mlflowListArtifactsResponse:

Response Structure
------------------






+------------+-----------------------------------+-------------------------------------------+
| Field Name |               Type                |                Description                |
+============+===================================+===========================================+
| root_uri   | ``STRING``                        | Root artifact directory for the run.      |
+------------+-----------------------------------+-------------------------------------------+
| files      | An array of :ref:`mlflowfileinfo` | File location and metadata for artifacts. |
+------------+-----------------------------------+-------------------------------------------+

===========================



.. _mlflowMlflowServiceupdateRun:

Update Run
==========


+----------------------------+-------------+
|          Endpoint          | HTTP Method |
+============================+=============+
| ``2.0/mlflow/runs/update`` | ``POST``    |
+----------------------------+-------------+

Update run metadata.




.. _mlflowUpdateRun:

Request Structure
-----------------






+------------+------------------------+----------------------------------------------------------------------------+
| Field Name |          Type          |                                Description                                 |
+============+========================+============================================================================+
| run_id     | ``STRING``             | ID of the run to update. Must be provided.                                 |
+------------+------------------------+----------------------------------------------------------------------------+
| run_uuid   | ``STRING``             | [Deprecated, use run_id instead] ID of the run to update.. This field will |
|            |                        | be removed in a future MLflow version.                                     |
+------------+------------------------+----------------------------------------------------------------------------+
| status     | :ref:`mlflowrunstatus` | Updated status of the run.                                                 |
+------------+------------------------+----------------------------------------------------------------------------+
| end_time   | ``INT64``              | Unix timestamp in milliseconds of when the run ended.                      |
+------------+------------------------+----------------------------------------------------------------------------+

.. _mlflowUpdateRunResponse:

Response Structure
------------------






+------------+----------------------+------------------------------+
| Field Name |         Type         |         Description          |
+============+======================+==============================+
| run_info   | :ref:`mlflowruninfo` | Updated metadata of the run. |
+------------+----------------------+------------------------------+

===========================



.. _mlflowModelRegistryServicecreateRegisteredModel:

Create RegisteredModel
======================


+-------------------------------------------------+-------------+
|                    Endpoint                     | HTTP Method |
+=================================================+=============+
| ``2.0/preview/mlflow/registered-models/create`` | ``POST``    |
+-------------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.

Throws ``RESOURCE_ALREADY_EXISTS`` if a registered model with the given name exists.




.. _mlflowCreateRegisteredModel:

Request Structure
-----------------






+------------+------------+---------------------------------+
| Field Name |    Type    |           Description           |
+============+============+=================================+
| name       | ``STRING`` | Register models under this name |
|            |            | This field is required.         |
|            |            |                                 |
+------------+------------+---------------------------------+

.. _mlflowCreateRegisteredModelResponse:

Response Structure
------------------






+------------------+------------------------------+-------------+
|    Field Name    |             Type             | Description |
+==================+==============================+=============+
| registered_model | :ref:`mlflowregisteredmodel` |             |
+------------------+------------------------------+-------------+

===========================



.. _mlflowModelRegistryServicegetRegisteredModelDetails:

Get RegisteredModel Details
===========================


+------------------------------------------------------+-------------+
|                       Endpoint                       | HTTP Method |
+======================================================+=============+
| ``2.0/preview/mlflow/registered-models/get-details`` | ``POST``    |
+------------------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowGetRegisteredModelDetails:

Request Structure
-----------------






+------------------+------------------------------+-------------------------+
|    Field Name    |             Type             |       Description       |
+==================+==============================+=========================+
| registered_model | :ref:`mlflowregisteredmodel` | Registered model.       |
|                  |                              | This field is required. |
|                  |                              |                         |
+------------------+------------------------------+-------------------------+

.. _mlflowGetRegisteredModelDetailsResponse:

Response Structure
------------------






+---------------------------+--------------------------------------+-------------+
|        Field Name         |                 Type                 | Description |
+===========================+======================================+=============+
| registered_model_detailed | :ref:`mlflowregisteredmodeldetailed` |             |
+---------------------------+--------------------------------------+-------------+

===========================



.. _mlflowModelRegistryServiceupdateRegisteredModel:

Update RegisteredModel
======================


+-------------------------------------------------+-------------+
|                    Endpoint                     | HTTP Method |
+=================================================+=============+
| ``2.0/preview/mlflow/registered-models/update`` | ``PATCH``   |
+-------------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowUpdateRegisteredModel:

Request Structure
-----------------






+------------------+------------------------------+---------------------------------------------------------------------+
|    Field Name    |             Type             |                             Description                             |
+==================+==============================+=====================================================================+
| registered_model | :ref:`mlflowregisteredmodel` | Registered model.                                                   |
|                  |                              | This field is required.                                             |
|                  |                              |                                                                     |
+------------------+------------------------------+---------------------------------------------------------------------+
| name             | ``STRING``                   | If provided, updates the name for this ``registered_model``.        |
+------------------+------------------------------+---------------------------------------------------------------------+
| description      | ``STRING``                   | If provided, updates the description for this ``registered_model``. |
+------------------+------------------------------+---------------------------------------------------------------------+

.. _mlflowUpdateRegisteredModelResponse:

Response Structure
------------------






+------------------+------------------------------+-------------+
|    Field Name    |             Type             | Description |
+==================+==============================+=============+
| registered_model | :ref:`mlflowregisteredmodel` |             |
+------------------+------------------------------+-------------+

===========================



.. _mlflowModelRegistryServicedeleteRegisteredModel:

Delete RegisteredModel
======================


+-------------------------------------------------+-------------+
|                    Endpoint                     | HTTP Method |
+=================================================+=============+
| ``2.0/preview/mlflow/registered-models/delete`` | ``DELETE``  |
+-------------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowDeleteRegisteredModel:

Request Structure
-----------------






+------------------+------------------------------+-------------------------+
|    Field Name    |             Type             |       Description       |
+==================+==============================+=========================+
| registered_model | :ref:`mlflowregisteredmodel` | Registered model.       |
|                  |                              | This field is required. |
|                  |                              |                         |
+------------------+------------------------------+-------------------------+

===========================



.. _mlflowModelRegistryServicelistRegisteredModels:

List RegisteredModels
=====================


+-----------------------------------------------+-------------+
|                   Endpoint                    | HTTP Method |
+===============================================+=============+
| ``2.0/preview/mlflow/registered-models/list`` | ``GET``     |
+-----------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowListRegisteredModelsResponse:

Response Structure
------------------






+----------------------------+--------------------------------------------------+-------------+
|         Field Name         |                       Type                       | Description |
+============================+==================================================+=============+
| registered_models_detailed | An array of :ref:`mlflowregisteredmodeldetailed` |             |
+----------------------------+--------------------------------------------------+-------------+

===========================



.. _mlflowModelRegistryServicegetLatestVersions:

Get Latest ModelVersions
========================


+--------------------------------------------------------------+-------------+
|                           Endpoint                           | HTTP Method |
+==============================================================+=============+
| ``2.0/preview/mlflow/registered-models/get-latest-versions`` | ``POST``    |
+--------------------------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowGetLatestVersions:

Request Structure
-----------------






+------------------+------------------------------+-------------------------+
|    Field Name    |             Type             |       Description       |
+==================+==============================+=========================+
| registered_model | :ref:`mlflowregisteredmodel` | Registered model.       |
|                  |                              | This field is required. |
|                  |                              |                         |
+------------------+------------------------------+-------------------------+
| stages           | An array of ``STRING``       | List of stages.         |
+------------------+------------------------------+-------------------------+

.. _mlflowGetLatestVersionsResponse:

Response Structure
------------------






+-------------------------+-----------------------------------------------+--------------------------------------------------------------------------------------------------+
|       Field Name        |                     Type                      |                                           Description                                            |
+=========================+===============================================+==================================================================================================+
| model_versions_detailed | An array of :ref:`mlflowmodelversiondetailed` | Latest version models for each requests stage. Only return models with current ``READY`` status. |
|                         |                                               | If no ``stages`` provided, returns the latest version for each stage, including ``"None"``.      |
+-------------------------+-----------------------------------------------+--------------------------------------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicecreateModelVersion:

Create ModelVersion
===================


+----------------------------------------------+-------------+
|                   Endpoint                   | HTTP Method |
+==============================================+=============+
| ``2.0/preview/mlflow/model-versions/create`` | ``POST``    |
+----------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowCreateModelVersion:

Request Structure
-----------------






+------------+------------+------------------------------------------------------------------------------------+
| Field Name |    Type    |                                    Description                                     |
+============+============+====================================================================================+
| name       | ``STRING`` | Register model under this name                                                     |
|            |            | This field is required.                                                            |
|            |            |                                                                                    |
+------------+------------+------------------------------------------------------------------------------------+
| source     | ``STRING`` | URI indicating the location of the model artifacts.                                |
|            |            | This field is required.                                                            |
|            |            |                                                                                    |
+------------+------------+------------------------------------------------------------------------------------+
| run_id     | ``STRING`` | MLflow run ID for correlation, if ``source`` was generated by an experiment run in |
|            |            | MLflow tracking server                                                             |
+------------+------------+------------------------------------------------------------------------------------+

.. _mlflowCreateModelVersionResponse:

Response Structure
------------------






+---------------+---------------------------+-----------------------------------------------------------------+
|  Field Name   |           Type            |                           Description                           |
+===============+===========================+=================================================================+
| model_version | :ref:`mlflowmodelversion` | Return new version number generated for this model in registry. |
+---------------+---------------------------+-----------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicegetModelVersionDetails:

Get ModelVersion Details
========================


+---------------------------------------------------+-------------+
|                     Endpoint                      | HTTP Method |
+===================================================+=============+
| ``2.0/preview/mlflow/model-versions/get-details`` | ``POST``    |
+---------------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowGetModelVersionDetails:

Request Structure
-----------------






+---------------+---------------------------+-------------------------+
|  Field Name   |           Type            |       Description       |
+===============+===========================+=========================+
| model_version | :ref:`mlflowmodelversion` | Model version.          |
|               |                           | This field is required. |
|               |                           |                         |
+---------------+---------------------------+-------------------------+

.. _mlflowGetModelVersionDetailsResponse:

Response Structure
------------------






+------------------------+-----------------------------------+-------------+
|       Field Name       |               Type                | Description |
+========================+===================================+=============+
| model_version_detailed | :ref:`mlflowmodelversiondetailed` |             |
+------------------------+-----------------------------------+-------------+

===========================



.. _mlflowModelRegistryServiceupdateModelVersion:

Update ModelVersion
===================


+----------------------------------------------+-------------+
|                   Endpoint                   | HTTP Method |
+==============================================+=============+
| ``2.0/preview/mlflow/model-versions/update`` | ``PATCH``   |
+----------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowUpdateModelVersion:

Request Structure
-----------------






+---------------+---------------------------+---------------------------------------------------------------------+
|  Field Name   |           Type            |                             Description                             |
+===============+===========================+=====================================================================+
| model_version | :ref:`mlflowmodelversion` | Model version.                                                      |
|               |                           | This field is required.                                             |
|               |                           |                                                                     |
+---------------+---------------------------+---------------------------------------------------------------------+
| stage         | ``STRING``                | If provided, transition ``model_version`` to new stage.             |
+---------------+---------------------------+---------------------------------------------------------------------+
| description   | ``STRING``                | If provided, updates the description for this ``registered_model``. |
+---------------+---------------------------+---------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicedeleteModelVersion:

Delete ModelVersion
===================


+----------------------------------------------+-------------+
|                   Endpoint                   | HTTP Method |
+==============================================+=============+
| ``2.0/preview/mlflow/model-versions/delete`` | ``DELETE``  |
+----------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowDeleteModelVersion:

Request Structure
-----------------






+---------------+---------------------------+-------------------------+
|  Field Name   |           Type            |       Description       |
+===============+===========================+=========================+
| model_version | :ref:`mlflowmodelversion` | Model version.          |
|               |                           | This field is required. |
|               |                           |                         |
+---------------+---------------------------+-------------------------+

===========================



.. _mlflowModelRegistryServicesearchModelVersions:

Search ModelVersions
====================


+----------------------------------------------+-------------+
|                   Endpoint                   | HTTP Method |
+==============================================+=============+
| ``2.0/preview/mlflow/model-versions/search`` | ``GET``     |
+----------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowSearchModelVersions:

Request Structure
-----------------






+-------------+------------------------+--------------------------------------------------------------------------------------------+
| Field Name  |          Type          |                                        Description                                         |
+=============+========================+============================================================================================+
| filter      | ``STRING``             | String filter condition, like "name='my-model-name'". Must be a single boolean condition,  |
|             |                        | with string values wrapped in single quotes.                                               |
+-------------+------------------------+--------------------------------------------------------------------------------------------+
| max_results | ``INT64``              | Maximum number of models desired. Max threshold is 1000.                                   |
+-------------+------------------------+--------------------------------------------------------------------------------------------+
| order_by    | An array of ``STRING`` | List of columns to be ordered by including model name, version, stage with an              |
|             |                        | optional "DESC" or "ASC" annotation, where "ASC" is the default.                           |
|             |                        | Tiebreaks are done by latest stage transition timestamp, followed by name ASC, followed by |
|             |                        | version DESC.                                                                              |
+-------------+------------------------+--------------------------------------------------------------------------------------------+
| page_token  | ``STRING``             | Pagination token to go to next page based on previous search query.                        |
+-------------+------------------------+--------------------------------------------------------------------------------------------+

.. _mlflowSearchModelVersionsResponse:

Response Structure
------------------






+-------------------------+-----------------------------------------------+----------------------------------------------------------------------------+
|       Field Name        |                     Type                      |                                Description                                 |
+=========================+===============================================+============================================================================+
| model_versions_detailed | An array of :ref:`mlflowmodelversiondetailed` | Models that match the search criteria                                      |
+-------------------------+-----------------------------------------------+----------------------------------------------------------------------------+
| next_page_token         | ``STRING``                                    | Pagination token to request next page of models for the same search query. |
+-------------------------+-----------------------------------------------+----------------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicegetModelVersionDownloadUri:

Get Download URI For ModelVersion Artifacts
===========================================


+--------------------------------------------------------+-------------+
|                        Endpoint                        | HTTP Method |
+========================================================+=============+
| ``2.0/preview/mlflow/model-versions/get-download-uri`` | ``POST``    |
+--------------------------------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowGetModelVersionDownloadUri:

Request Structure
-----------------






+---------------+---------------------------+---------------------------+
|  Field Name   |           Type            |        Description        |
+===============+===========================+===========================+
| model_version | :ref:`mlflowmodelversion` | Name and version of model |
|               |                           | This field is required.   |
|               |                           |                           |
+---------------+---------------------------+---------------------------+

.. _mlflowGetModelVersionDownloadUriResponse:

Response Structure
------------------






+--------------+------------+-------------------------------------------------------------------------+
|  Field Name  |    Type    |                               Description                               |
+==============+============+=========================================================================+
| artifact_uri | ``STRING`` | URI corresponding to where artifacts for this model version are stored. |
+--------------+------------+-------------------------------------------------------------------------+

.. _RESTadd:

Data Structures
===============



.. _mlflowExperiment:

Experiment
----------



Experiment


+-------------------+----------------------------------------+--------------------------------------------------------------------+
|    Field Name     |                  Type                  |                            Description                             |
+===================+========================================+====================================================================+
| experiment_id     | ``STRING``                             | Unique identifier for the experiment.                              |
+-------------------+----------------------------------------+--------------------------------------------------------------------+
| name              | ``STRING``                             | Human readable name that identifies the experiment.                |
+-------------------+----------------------------------------+--------------------------------------------------------------------+
| artifact_location | ``STRING``                             | Location where artifacts for the experiment are stored.            |
+-------------------+----------------------------------------+--------------------------------------------------------------------+
| lifecycle_stage   | ``STRING``                             | Current life cycle stage of the experiment: "active" or "deleted". |
|                   |                                        | Deleted experiments are not returned by APIs.                      |
+-------------------+----------------------------------------+--------------------------------------------------------------------+
| last_update_time  | ``INT64``                              | Last update time                                                   |
+-------------------+----------------------------------------+--------------------------------------------------------------------+
| creation_time     | ``INT64``                              | Creation time                                                      |
+-------------------+----------------------------------------+--------------------------------------------------------------------+
| tags              | An array of :ref:`mlflowexperimenttag` | Tags: Additional metadata key-value pairs.                         |
+-------------------+----------------------------------------+--------------------------------------------------------------------+

.. _mlflowExperimentTag:

ExperimentTag
-------------



Tag for an experiment.


+------------+------------+----------------+
| Field Name |    Type    |  Description   |
+============+============+================+
| key        | ``STRING`` | The tag key.   |
+------------+------------+----------------+
| value      | ``STRING`` | The tag value. |
+------------+------------+----------------+

.. _mlflowFileInfo:

FileInfo
--------



Metadata of a single artifact file or directory.


+------------+------------+---------------------------------------------------+
| Field Name |    Type    |                    Description                    |
+============+============+===================================================+
| path       | ``STRING`` | Path relative to the root artifact directory run. |
+------------+------------+---------------------------------------------------+
| is_dir     | ``BOOL``   | Whether the path is a directory.                  |
+------------+------------+---------------------------------------------------+
| file_size  | ``INT64``  | Size in bytes. Unset for directories.             |
+------------+------------+---------------------------------------------------+

.. _mlflowMetric:

Metric
------



Metric associated with a run, represented as a key-value pair.


+------------+------------+--------------------------------------------------+
| Field Name |    Type    |                   Description                    |
+============+============+==================================================+
| key        | ``STRING`` | Key identifying this metric.                     |
+------------+------------+--------------------------------------------------+
| value      | ``DOUBLE`` | Value associated with this metric.               |
+------------+------------+--------------------------------------------------+
| timestamp  | ``INT64``  | The timestamp at which this metric was recorded. |
+------------+------------+--------------------------------------------------+
| step       | ``INT64``  | Step at which to log the metric.                 |
+------------+------------+--------------------------------------------------+

.. _mlflowModelVersion:

ModelVersion
------------



.. note::
    Experimental: This entity may change or be removed in a future release without warning.


+------------------+------------------------------+-------------------------+
|    Field Name    |             Type             |       Description       |
+==================+==============================+=========================+
| registered_model | :ref:`mlflowregisteredmodel` | Registered model.       |
+------------------+------------------------------+-------------------------+
| version          | ``INT64``                    | Model's version number. |
+------------------+------------------------------+-------------------------+

.. _mlflowModelVersionDetailed:

ModelVersionDetailed
--------------------



.. note::
    Experimental: This entity may change or be removed in a future release without warning.


+------------------------+---------------------------------+-------------------------------------------------------------------------------------------------+
|       Field Name       |              Type               |                                           Description                                           |
+========================+=================================+=================================================================================================+
| model_version          | :ref:`mlflowmodelversion`       | Model Version                                                                                   |
+------------------------+---------------------------------+-------------------------------------------------------------------------------------------------+
| creation_timestamp     | ``INT64``                       | Timestamp recorded when this ``model_version`` was created.                                     |
+------------------------+---------------------------------+-------------------------------------------------------------------------------------------------+
| last_updated_timestamp | ``INT64``                       | Timestamp recorded when metadata for this ``model_version`` was last updated.                   |
+------------------------+---------------------------------+-------------------------------------------------------------------------------------------------+
| user_id                | ``STRING``                      | User that created this ``model_version``.                                                       |
+------------------------+---------------------------------+-------------------------------------------------------------------------------------------------+
| current_stage          | ``STRING``                      | Current stage for this ``model_version``.                                                       |
+------------------------+---------------------------------+-------------------------------------------------------------------------------------------------+
| description            | ``STRING``                      | Description of this ``model_version``.                                                          |
+------------------------+---------------------------------+-------------------------------------------------------------------------------------------------+
| source                 | ``STRING``                      | URI indicating the location of the source model artifacts, used when creating ``model_version`` |
+------------------------+---------------------------------+-------------------------------------------------------------------------------------------------+
| run_id                 | ``STRING``                      | MLflow run ID used when creating ``model_version``, if ``source`` was generated by an           |
|                        |                                 | experiment run stored in MLflow tracking server.                                                |
+------------------------+---------------------------------+-------------------------------------------------------------------------------------------------+
| status                 | :ref:`mlflowmodelversionstatus` | Current status of ``model_version``                                                             |
+------------------------+---------------------------------+-------------------------------------------------------------------------------------------------+
| status_message         | ``STRING``                      | Details on current ``status``, if it is pending or failed.                                      |
+------------------------+---------------------------------+-------------------------------------------------------------------------------------------------+

.. _mlflowParam:

Param
-----



Param associated with a run.


+------------+------------+-----------------------------------+
| Field Name |    Type    |            Description            |
+============+============+===================================+
| key        | ``STRING`` | Key identifying this param.       |
+------------+------------+-----------------------------------+
| value      | ``STRING`` | Value associated with this param. |
+------------+------------+-----------------------------------+

.. _mlflowRegisteredModel:

RegisteredModel
---------------



.. note::
    Experimental: This entity may change or be removed in a future release without warning.


+------------+------------+----------------------------+
| Field Name |    Type    |        Description         |
+============+============+============================+
| name       | ``STRING`` | Unique name for the model. |
+------------+------------+----------------------------+

.. _mlflowRegisteredModelDetailed:

RegisteredModelDetailed
-----------------------



.. note::
    Experimental: This entity may change or be removed in a future release without warning.


+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
|       Field Name       |                     Type                      |                                   Description                                    |
+========================+===============================================+==================================================================================+
| registered_model       | :ref:`mlflowregisteredmodel`                  | Registered model.                                                                |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| creation_timestamp     | ``INT64``                                     | Timestamp recorded when this ``registered_model`` was created.                   |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| last_updated_timestamp | ``INT64``                                     | Timestamp recorded when metadata for this ``registered_model`` was last updated. |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| user_id                | ``STRING``                                    | User that created this ``registered_model``                                      |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| description            | ``STRING``                                    | Description of this ``registered_model``.                                        |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| latest_versions        | An array of :ref:`mlflowmodelversiondetailed` | Collection of latest model versions for each stage.                              |
|                        |                                               | Only contains models with current ``READY`` status.                              |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+

.. _mlflowRun:

Run
---



A single run.


+------------+----------------------+---------------+
| Field Name |         Type         |  Description  |
+============+======================+===============+
| info       | :ref:`mlflowruninfo` | Run metadata. |
+------------+----------------------+---------------+
| data       | :ref:`mlflowrundata` | Run data.     |
+------------+----------------------+---------------+

.. _mlflowRunData:

RunData
-------



Run data (metrics, params, and tags).


+------------+---------------------------------+--------------------------------------+
| Field Name |              Type               |             Description              |
+============+=================================+======================================+
| metrics    | An array of :ref:`mlflowmetric` | Run metrics.                         |
+------------+---------------------------------+--------------------------------------+
| params     | An array of :ref:`mlflowparam`  | Run parameters.                      |
+------------+---------------------------------+--------------------------------------+
| tags       | An array of :ref:`mlflowruntag` | Additional metadata key-value pairs. |
+------------+---------------------------------+--------------------------------------+

.. _mlflowRunInfo:

RunInfo
-------



Metadata of a single run.


+-----------------+------------------------+----------------------------------------------------------------------------------+
|   Field Name    |          Type          |                                   Description                                    |
+=================+========================+==================================================================================+
| run_id          | ``STRING``             | Unique identifier for the run.                                                   |
+-----------------+------------------------+----------------------------------------------------------------------------------+
| run_uuid        | ``STRING``             | [Deprecated, use run_id instead] Unique identifier for the run. This field will  |
|                 |                        | be removed in a future MLflow version.                                           |
+-----------------+------------------------+----------------------------------------------------------------------------------+
| experiment_id   | ``STRING``             | The experiment ID.                                                               |
+-----------------+------------------------+----------------------------------------------------------------------------------+
| user_id         | ``STRING``             | User who initiated the run.                                                      |
|                 |                        | This field is deprecated as of MLflow 1.0, and will be removed in a future       |
|                 |                        | MLflow release. Use 'mlflow.user' tag instead.                                   |
+-----------------+------------------------+----------------------------------------------------------------------------------+
| status          | :ref:`mlflowrunstatus` | Current status of the run.                                                       |
+-----------------+------------------------+----------------------------------------------------------------------------------+
| start_time      | ``INT64``              | Unix timestamp of when the run started in milliseconds.                          |
+-----------------+------------------------+----------------------------------------------------------------------------------+
| end_time        | ``INT64``              | Unix timestamp of when the run ended in milliseconds.                            |
+-----------------+------------------------+----------------------------------------------------------------------------------+
| artifact_uri    | ``STRING``             | URI of the directory where artifacts should be uploaded.                         |
|                 |                        | This can be a local path (starting with "/"), or a distributed file system (DFS) |
|                 |                        | path, like ``s3://bucket/directory`` or ``dbfs:/my/directory``.                  |
|                 |                        | If not set, the local ``./mlruns`` directory is  chosen.                         |
+-----------------+------------------------+----------------------------------------------------------------------------------+
| lifecycle_stage | ``STRING``             | Current life cycle stage of the experiment : OneOf("active", "deleted")          |
+-----------------+------------------------+----------------------------------------------------------------------------------+

.. _mlflowRunTag:

RunTag
------



Tag for a run.


+------------+------------+----------------+
| Field Name |    Type    |  Description   |
+============+============+================+
| key        | ``STRING`` | The tag key.   |
+------------+------------+----------------+
| value      | ``STRING`` | The tag value. |
+------------+------------+----------------+

.. _mlflowModelVersionStatus:

ModelVersionStatus
------------------


.. note::
    Experimental: This entity may change or be removed in a future release without warning.

+----------------------+---------------------------------------------------------------------------------------------+
|         Name         |                                         Description                                         |
+======================+=============================================================================================+
| PENDING_REGISTRATION | Request to register a new model version is pending as server performs background tasks.     |
+----------------------+---------------------------------------------------------------------------------------------+
| FAILED_REGISTRATION  | Request to register a new model version has failed.                                         |
+----------------------+---------------------------------------------------------------------------------------------+
| READY                | Model version is ready for use.                                                             |
+----------------------+---------------------------------------------------------------------------------------------+
| PENDING_DELETION     | Request to delete an existing model version is pending as server performs background tasks. |
+----------------------+---------------------------------------------------------------------------------------------+
| FAILED_DELETION      | Request to delete an existing model version has failed.                                     |
+----------------------+---------------------------------------------------------------------------------------------+

.. _mlflowRunStatus:

RunStatus
---------


Status of a run.

+-----------+------------------------------------------+
|   Name    |               Description                |
+===========+==========================================+
| RUNNING   | Run has been initiated.                  |
+-----------+------------------------------------------+
| SCHEDULED | Run is scheduled to run at a later time. |
+-----------+------------------------------------------+
| FINISHED  | Run has completed.                       |
+-----------+------------------------------------------+
| FAILED    | Run execution failed.                    |
+-----------+------------------------------------------+
| KILLED    | Run killed by user.                      |
+-----------+------------------------------------------+

.. _mlflowViewType:

ViewType
--------


View type for ListExperiments query.

+--------------+------------------------------------------+
|     Name     |               Description                |
+==============+==========================================+
| ACTIVE_ONLY  | Default. Return only active experiments. |
+--------------+------------------------------------------+
| DELETED_ONLY | Return only deleted experiments.         |
+--------------+------------------------------------------+
| ALL          | Get all experiments.                     |
+--------------+------------------------------------------+