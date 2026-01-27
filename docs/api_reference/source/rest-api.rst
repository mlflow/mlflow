
.. _rest-api:

========
REST API
========


The MLflow REST API allows you to create, list, and get experiments and runs, and log
parameters, metrics, and artifacts. The API is hosted under the ``/api`` route on the MLflow
tracking server. For example, to search for experiments on a tracking server hosted at
``http://localhost:5000``, make a POST request to ``http://localhost:5000/api/2.0/mlflow/experiments/search``.

.. important::
    The MLflow REST API requires content type ``application/json`` for all POST requests.

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






+-------------------+----------------------------------------+------------------------------------------------------------------------------------------------+
|    Field Name     |                  Type                  |                                          Description                                           |
+===================+========================================+================================================================================================+
| name              | ``STRING``                             | Experiment name.                                                                               |
|                   |                                        | This field is required.                                                                        |
|                   |                                        |                                                                                                |
+-------------------+----------------------------------------+------------------------------------------------------------------------------------------------+
| artifact_location | ``STRING``                             | Location where all artifacts for the experiment are stored.                                    |
|                   |                                        | If not provided, the remote server will select an appropriate default.                         |
+-------------------+----------------------------------------+------------------------------------------------------------------------------------------------+
| tags              | An array of :ref:`mlflowexperimenttag` | A collection of tags to set on the experiment. Maximum tag size and number of tags per request |
|                   |                                        | depends on the storage backend. All storage backends are guaranteed to support tag keys up     |
|                   |                                        | to 250 bytes in size and tag values up to 5000 bytes in size. All storage backends are also    |
|                   |                                        | guaranteed to support up to 20 tags per request.                                               |
+-------------------+----------------------------------------+------------------------------------------------------------------------------------------------+

.. _mlflowCreateExperimentResponse:

Response Structure
------------------






+---------------+------------+---------------------------------------+
|  Field Name   |    Type    |              Description              |
+===============+============+=======================================+
| experiment_id | ``STRING`` | Unique identifier for the experiment. |
+---------------+------------+---------------------------------------+

===========================



.. _mlflowMlflowServicesearchExperiments:

Search Experiments
==================


+-----------------------------------+-------------+
|             Endpoint              | HTTP Method |
+===================================+=============+
| ``2.0/mlflow/experiments/search`` | ``POST``    |
+-----------------------------------+-------------+






.. _mlflowSearchExperiments:

Request Structure
-----------------






+-------------+------------------------+--------------------------------------------------------------------------------------------+
| Field Name  |          Type          |                                        Description                                         |
+=============+========================+============================================================================================+
| max_results | ``INT64``              | Maximum number of experiments desired.                                                     |
|             |                        | Servers may select a desired default `max_results` value. All servers are                  |
|             |                        | guaranteed to support a `max_results` threshold of at least 1,000 but may                  |
|             |                        | support more. Callers of this endpoint are encouraged to pass max_results                  |
|             |                        | explicitly and leverage page_token to iterate through experiments.                         |
+-------------+------------------------+--------------------------------------------------------------------------------------------+
| page_token  | ``STRING``             | Token indicating the page of experiments to fetch                                          |
+-------------+------------------------+--------------------------------------------------------------------------------------------+
| filter      | ``STRING``             | A filter expression over experiment attributes and tags that allows returning a subset of  |
|             |                        | experiments. The syntax is a subset of SQL that supports ANDing together binary operations |
|             |                        | between an attribute or tag, and a constant.                                               |
|             |                        |                                                                                            |
|             |                        | Example: ``name LIKE 'test-%' AND tags.key = 'value'``                                     |
|             |                        |                                                                                            |
|             |                        | You can select columns with special characters (hyphen, space, period, etc.) by using      |
|             |                        | double quotes or backticks.                                                                |
|             |                        |                                                                                            |
|             |                        | Example: ``tags."extra-key" = 'value'`` or ``tags.`extra-key` = 'value'``                  |
|             |                        |                                                                                            |
|             |                        | Supported operators are ``=``, ``!=``, ``LIKE``, and ``ILIKE``.                            |
+-------------+------------------------+--------------------------------------------------------------------------------------------+
| order_by    | An array of ``STRING`` | List of columns for ordering search results, which can include experiment name and id      |
|             |                        | with an optional "DESC" or "ASC" annotation, where "ASC" is the default.                   |
|             |                        | Tiebreaks are done by experiment id DESC.                                                  |
+-------------+------------------------+--------------------------------------------------------------------------------------------+
| view_type   | :ref:`mlflowviewtype`  | Qualifier for type of experiments to be returned.                                          |
|             |                        | If unspecified, return only active experiments.                                            |
+-------------+------------------------+--------------------------------------------------------------------------------------------+

.. _mlflowSearchExperimentsResponse:

Response Structure
------------------






+-----------------+-------------------------------------+----------------------------------------------------------------------------+
|   Field Name    |                Type                 |                                Description                                 |
+=================+=====================================+============================================================================+
| experiments     | An array of :ref:`mlflowexperiment` | Experiments that match the search criteria                                 |
+-----------------+-------------------------------------+----------------------------------------------------------------------------+
| next_page_token | ``STRING``                          | Token that can be used to retrieve the next page of experiments.           |
|                 |                                     | An empty token means that no more experiments are available for retrieval. |
+-----------------+-------------------------------------+----------------------------------------------------------------------------+

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






+------------+-------------------------+---------------------+
| Field Name |          Type           |     Description     |
+============+=========================+=====================+
| experiment | :ref:`mlflowexperiment` | Experiment details. |
+------------+-------------------------+---------------------+

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



.. _mlflowMlflowServicedeleteExperimentTag:

Delete Experiment Tag
=====================


+--------------------------------------------------+-------------+
|                     Endpoint                     | HTTP Method |
+==================================================+=============+
| ``2.0/mlflow/experiments/delete-experiment-tag`` | ``POST``    |
+--------------------------------------------------+-------------+

Delete a tag on an experiment.




.. _mlflowDeleteExperimentTag:

Request Structure
-----------------






+---------------+------------+-----------------------------------------------------------------------+
|  Field Name   |    Type    |                              Description                              |
+===============+============+=======================================================================+
| experiment_id | ``STRING`` | ID of the experiment that the tag was logged under. Must be provided. |
|               |            | This field is required.                                               |
|               |            |                                                                       |
+---------------+------------+-----------------------------------------------------------------------+
| key           | ``STRING`` | Name of the tag. Maximum size is 255 bytes. Must be provided.         |
|               |            | This field is required.                                               |
|               |            |                                                                       |
+---------------+------------+-----------------------------------------------------------------------+

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
| run_name      | ``STRING``                      | Name of the run.                                                           |
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
| run_name   | ``STRING``             | Updated name of the run.                                                   |
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
| max_results    | ``INT32``              | Maximum number of runs desired. If unspecified, defaults to 1000.                                    |
|                |                        | All servers are guaranteed to support a `max_results` threshold of at least 50,000                   |
|                |                        | but may support more. Callers of this endpoint are encouraged to pass max_results                    |
|                |                        | explicitly and leverage page_token to iterate through experiments.                                   |
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






+----------------+------------+-----------------------------------------------------------------------------------------------+
|   Field Name   |    Type    |                                          Description                                          |
+================+============+===============================================================================================+
| run_id         | ``STRING`` | ID of the run under which to log the metric. Must be provided.                                |
+----------------+------------+-----------------------------------------------------------------------------------------------+
| run_uuid       | ``STRING`` | [Deprecated, use run_id instead] ID of the run under which to log the metric. This field will |
|                |            | be removed in a future MLflow version.                                                        |
+----------------+------------+-----------------------------------------------------------------------------------------------+
| key            | ``STRING`` | Name of the metric.                                                                           |
|                |            | This field is required.                                                                       |
|                |            |                                                                                               |
+----------------+------------+-----------------------------------------------------------------------------------------------+
| value          | ``DOUBLE`` | Double value of the metric being logged.                                                      |
|                |            | This field is required.                                                                       |
|                |            |                                                                                               |
+----------------+------------+-----------------------------------------------------------------------------------------------+
| timestamp      | ``INT64``  | Unix timestamp in milliseconds at the time metric was logged.                                 |
|                |            | This field is required.                                                                       |
|                |            |                                                                                               |
+----------------+------------+-----------------------------------------------------------------------------------------------+
| step           | ``INT64``  | Step at which to log the metric                                                               |
+----------------+------------+-----------------------------------------------------------------------------------------------+
| model_id       | ``STRING`` | ID of the logged model associated with the metric, if applicable                              |
+----------------+------------+-----------------------------------------------------------------------------------------------+
| dataset_name   | ``STRING`` | The name of the dataset associated with the metric.                                           |
|                |            | E.g. "my.uc.table@2" "nyc-taxi-dataset", "fantastic-elk-3"                                    |
+----------------+------------+-----------------------------------------------------------------------------------------------+
| dataset_digest | ``STRING`` | Dataset digest of the dataset associated with the metric,                                     |
|                |            | e.g. an md5 hash of the dataset that uniquely identifies it                                   |
|                |            | within datasets of the same name.                                                             |
+----------------+------------+-----------------------------------------------------------------------------------------------+

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
| value      | ``STRING`` | String value of the param being logged. Maximum size is 6000 bytes.                          |
|            |            | This field is required.                                                                      |
|            |            |                                                                                              |
+------------+------------+----------------------------------------------------------------------------------------------+

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



.. _mlflowMlflowServicelogModel:

Log Model
=========


+-------------------------------+-------------+
|           Endpoint            | HTTP Method |
+===============================+=============+
| ``2.0/mlflow/runs/log-model`` | ``POST``    |
+-------------------------------+-------------+

.. note::
    Experimental: This API may change or be removed in a future release without warning.




.. _mlflowLogModel:

Request Structure
-----------------






+------------+------------+------------------------------+
| Field Name |    Type    |         Description          |
+============+============+==============================+
| run_id     | ``STRING`` | ID of the run to log under   |
+------------+------------+------------------------------+
| model_json | ``STRING`` | MLmodel file in json format. |
+------------+------------+------------------------------+

===========================



.. _mlflowMlflowServicelogInputs:

Log Inputs
==========


+--------------------------------+-------------+
|            Endpoint            | HTTP Method |
+================================+=============+
| ``2.0/mlflow/runs/log-inputs`` | ``POST``    |
+--------------------------------+-------------+






.. _mlflowLogInputs:

Request Structure
-----------------






+------------+---------------------------------------+-----------------------------------------------------------+
| Field Name |                 Type                  |                        Description                        |
+============+=======================================+===========================================================+
| run_id     | ``STRING``                            | ID of the run to log under                                |
|            |                                       | This field is required.                                   |
|            |                                       |                                                           |
+------------+---------------------------------------+-----------------------------------------------------------+
| datasets   | An array of :ref:`mlflowdatasetinput` | Dataset inputs                                            |
+------------+---------------------------------------+-----------------------------------------------------------+
| models     | An array of :ref:`mlflowmodelinput`   | Model inputs                                              |
|            |                                       | (Currently undocumented for LoggedModels private preview) |
+------------+---------------------------------------+-----------------------------------------------------------+

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






+-------------+------------+------------------------------------------------------------------------------------------------+
| Field Name  |    Type    |                                          Description                                           |
+=============+============+================================================================================================+
| run_id      | ``STRING`` | ID of the run from which to fetch metric values. Must be provided.                             |
+-------------+------------+------------------------------------------------------------------------------------------------+
| run_uuid    | ``STRING`` | [Deprecated, use run_id instead] ID of the run from which to fetch metric values. This field   |
|             |            | will be removed in a future MLflow version.                                                    |
+-------------+------------+------------------------------------------------------------------------------------------------+
| metric_key  | ``STRING`` | Name of the metric.                                                                            |
|             |            | This field is required.                                                                        |
|             |            |                                                                                                |
+-------------+------------+------------------------------------------------------------------------------------------------+
| page_token  | ``STRING`` | Token indicating the page of metric history to fetch                                           |
+-------------+------------+------------------------------------------------------------------------------------------------+
| max_results | ``INT32``  | Maximum number of logged instances of a metric for a run to return per call.                   |
|             |            | Backend servers may restrict the value of `max_results` depending on performance requirements. |
|             |            | Requests that do not specify this value will behave as non-paginated queries where all         |
|             |            | metric history values for a given metric within a run are returned in a single response.       |
+-------------+------------+------------------------------------------------------------------------------------------------+

.. _mlflowGetMetricHistoryResponse:

Response Structure
------------------






+-----------------+---------------------------------+-------------------------------------------------------------------------------------+
|   Field Name    |              Type               |                                     Description                                     |
+=================+=================================+=====================================================================================+
| metrics         | An array of :ref:`mlflowmetric` | All logged values for this metric.                                                  |
+-----------------+---------------------------------+-------------------------------------------------------------------------------------+
| next_page_token | ``STRING``                      | Token that can be used to issue a query for the next page of metric history values. |
|                 |                                 | A missing token indicates that no additional metrics are available to fetch.        |
+-----------------+---------------------------------+-------------------------------------------------------------------------------------+

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
| page_token | ``STRING`` | Token indicating the page of artifact results to fetch                                  |
+------------+------------+-----------------------------------------------------------------------------------------+

.. _mlflowListArtifactsResponse:

Response Structure
------------------






+-----------------+-----------------------------------+----------------------------------------------------------------------+
|   Field Name    |               Type                |                             Description                              |
+=================+===================================+======================================================================+
| root_uri        | ``STRING``                        | Root artifact directory for the run.                                 |
+-----------------+-----------------------------------+----------------------------------------------------------------------+
| files           | An array of :ref:`mlflowfileinfo` | File location and metadata for artifacts.                            |
+-----------------+-----------------------------------+----------------------------------------------------------------------+
| next_page_token | ``STRING``                        | Token that can be used to retrieve the next page of artifact results |
+-----------------+-----------------------------------+----------------------------------------------------------------------+

===========================



.. _mlflowMlflowServiceregisterScorer:

Register Scorer
===============


+---------------------------------+-------------+
|            Endpoint             | HTTP Method |
+=================================+=============+
| ``2.0/mlflow/scorers/register`` | ``POST``    |
+---------------------------------+-------------+

Register a scorer for an experiment.




.. _mlflowRegisterScorer:

Request Structure
-----------------



Register a scorer for an experiment.


+-------------------+------------+--------------------------------------+
|    Field Name     |    Type    |             Description              |
+===================+============+======================================+
| experiment_id     | ``STRING`` | The experiment ID.                   |
+-------------------+------------+--------------------------------------+
| name              | ``STRING`` | The scorer name.                     |
+-------------------+------------+--------------------------------------+
| serialized_scorer | ``STRING`` | The serialized scorer string (JSON). |
+-------------------+------------+--------------------------------------+

.. _mlflowRegisterScorerResponse:

Response Structure
------------------






+-------------------+------------+------------------------------------------------------------------------+
|    Field Name     |    Type    |                              Description                               |
+===================+============+========================================================================+
| version           | ``INT32``  | The new version number for the scorer.                                 |
+-------------------+------------+------------------------------------------------------------------------+
| scorer_id         | ``STRING`` | The unique identifier for the scorer.                                  |
+-------------------+------------+------------------------------------------------------------------------+
| experiment_id     | ``STRING`` | The experiment ID (same as request).                                   |
+-------------------+------------+------------------------------------------------------------------------+
| name              | ``STRING`` | The scorer name (same as request).                                     |
+-------------------+------------+------------------------------------------------------------------------+
| serialized_scorer | ``STRING`` | The serialized scorer string (same as request).                        |
+-------------------+------------+------------------------------------------------------------------------+
| creation_time     | ``INT64``  | The creation time of the scorer version (in milliseconds since epoch). |
+-------------------+------------+------------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicelistScorers:

List Scorers
============


+-----------------------------+-------------+
|          Endpoint           | HTTP Method |
+=============================+=============+
| ``2.0/mlflow/scorers/list`` | ``GET``     |
+-----------------------------+-------------+

List all scorers for an experiment.




.. _mlflowListScorers:

Request Structure
-----------------



List all scorers for an experiment.


+---------------+------------+--------------------+
|  Field Name   |    Type    |    Description     |
+===============+============+====================+
| experiment_id | ``STRING`` | The experiment ID. |
+---------------+------------+--------------------+

.. _mlflowListScorersResponse:

Response Structure
------------------






+------------+---------------------------------+----------------------------------------------------------------+
| Field Name |              Type               |                          Description                           |
+============+=================================+================================================================+
| scorers    | An array of :ref:`mlflowscorer` | List of scorer entities (latest version for each scorer name). |
+------------+---------------------------------+----------------------------------------------------------------+

===========================



.. _mlflowMlflowServicelistScorerVersions:

List Scorer Versions
====================


+---------------------------------+-------------+
|            Endpoint             | HTTP Method |
+=================================+=============+
| ``2.0/mlflow/scorers/versions`` | ``GET``     |
+---------------------------------+-------------+

List all versions of a specific scorer for an experiment.




.. _mlflowListScorerVersions:

Request Structure
-----------------



List all versions of a specific scorer for an experiment.


+---------------+------------+--------------------+
|  Field Name   |    Type    |    Description     |
+===============+============+====================+
| experiment_id | ``STRING`` | The experiment ID. |
+---------------+------------+--------------------+
| name          | ``STRING`` | The scorer name.   |
+---------------+------------+--------------------+

.. _mlflowListScorerVersionsResponse:

Response Structure
------------------






+------------+---------------------------------+---------------------------------------------------------+
| Field Name |              Type               |                       Description                       |
+============+=================================+=========================================================+
| scorers    | An array of :ref:`mlflowscorer` | List of scorer entities for all versions of the scorer. |
+------------+---------------------------------+---------------------------------------------------------+

===========================



.. _mlflowMlflowServicegetScorer:

Get Scorer
==========


+----------------------------+-------------+
|          Endpoint          | HTTP Method |
+============================+=============+
| ``2.0/mlflow/scorers/get`` | ``GET``     |
+----------------------------+-------------+

Get a specific scorer for an experiment.




.. _mlflowGetScorer:

Request Structure
-----------------



Get a specific scorer for an experiment.


+---------------+------------+--------------------------------------------------------------------------------+
|  Field Name   |    Type    |                                  Description                                   |
+===============+============+================================================================================+
| experiment_id | ``STRING`` | The experiment ID.                                                             |
+---------------+------------+--------------------------------------------------------------------------------+
| name          | ``STRING`` | The scorer name.                                                               |
+---------------+------------+--------------------------------------------------------------------------------+
| version       | ``INT32``  | The scorer version. If not specified, returns the scorer with maximum version. |
+---------------+------------+--------------------------------------------------------------------------------+

.. _mlflowGetScorerResponse:

Response Structure
------------------






+------------+---------------------+--------------------+
| Field Name |        Type         |    Description     |
+============+=====================+====================+
| scorer     | :ref:`mlflowscorer` | The scorer entity. |
+------------+---------------------+--------------------+

===========================



.. _mlflowMlflowServicedeleteScorer:

Delete Scorer
=============


+-------------------------------+-------------+
|           Endpoint            | HTTP Method |
+===============================+=============+
| ``2.0/mlflow/scorers/delete`` | ``DELETE``  |
+-------------------------------+-------------+

Delete a scorer for an experiment.




.. _mlflowDeleteScorer:

Request Structure
-----------------



Delete a scorer for an experiment.


+---------------+------------+-----------------------------------------------------------------------+
|  Field Name   |    Type    |                              Description                              |
+===============+============+=======================================================================+
| experiment_id | ``STRING`` | The experiment ID.                                                    |
+---------------+------------+-----------------------------------------------------------------------+
| name          | ``STRING`` | The scorer name.                                                      |
+---------------+------------+-----------------------------------------------------------------------+
| version       | ``INT32``  | The scorer version to delete. If not specified, deletes all versions. |
+---------------+------------+-----------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicecreateGatewaySecret:

Create Gateway Secret
=====================


+---------------------------------------+-------------+
|               Endpoint                | HTTP Method |
+=======================================+=============+
| ``2.0/mlflow/gateway/secrets/create`` | ``POST``    |
+---------------------------------------+-------------+

Create a new encrypted secret for LLM provider authentication




.. _mlflowCreateGatewaySecret:

Request Structure
-----------------






+--------------+--------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
|  Field Name  |                             Type                             |                                                      Description                                                       |
+==============+==============================================================+========================================================================================================================+
| secret_name  | ``STRING``                                                   | User-friendly name for the secret (must be unique)                                                                     |
+--------------+--------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| secret_value | An array of :ref:`mlflowcreategatewaysecretsecretvalueentry` | The secret value(s) to encrypt as key-value pairs.                                                                     |
|              |                                                              | For simple API keys: {"api_key": "sk-xxx"}                                                                             |
|              |                                                              | For compound credentials: {"aws_access_key_id": "...", "aws_secret_access_key": "..."}                                 |
+--------------+--------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| provider     | ``STRING``                                                   | Optional LLM provider (e.g., "openai", "anthropic")                                                                    |
+--------------+--------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| auth_config  | An array of :ref:`mlflowcreategatewaysecretauthconfigentry`  | Optional provider-specific auth configuration.                                                                         |
|              |                                                              | For multi-auth providers, include "auth_mode" key (e.g., {"auth_mode": "access_keys", "aws_region_name": "us-east-1"}) |
+--------------+--------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| created_by   | ``STRING``                                                   | Username of the creator                                                                                                |
+--------------+--------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+

.. _mlflowCreateGatewaySecretResponse:

Response Structure
------------------






+------------+--------------------------------+----------------------------------------------------------------+
| Field Name |              Type              |                          Description                           |
+============+================================+================================================================+
| secret     | :ref:`mlflowgatewaysecretinfo` | The created secret metadata (does not include encrypted value) |
+------------+--------------------------------+----------------------------------------------------------------+

===========================



.. _mlflowMlflowServicegetGatewaySecretInfo:

Get Gateway Secret Info
=======================


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/mlflow/gateway/secrets/get`` | ``GET``     |
+------------------------------------+-------------+

Get metadata about a secret (does not include the encrypted value)




.. _mlflowGetGatewaySecretInfo:

Request Structure
-----------------






+-------------+------------+--------------------------------------------------+
| Field Name  |    Type    |                   Description                    |
+=============+============+==================================================+
| secret_id   | ``STRING`` | Either secret_id or secret_name must be provided |
+-------------+------------+--------------------------------------------------+
| secret_name | ``STRING`` |                                                  |
+-------------+------------+--------------------------------------------------+

.. _mlflowGetGatewaySecretInfoResponse:

Response Structure
------------------






+------------+--------------------------------+----------------------------------------------------+
| Field Name |              Type              |                    Description                     |
+============+================================+====================================================+
| secret     | :ref:`mlflowgatewaysecretinfo` | Secret metadata (does not include encrypted value) |
+------------+--------------------------------+----------------------------------------------------+

===========================



.. _mlflowMlflowServiceupdateGatewaySecret:

Update Gateway Secret
=====================


+---------------------------------------+-------------+
|               Endpoint                | HTTP Method |
+=======================================+=============+
| ``2.0/mlflow/gateway/secrets/update`` | ``POST``    |
+---------------------------------------+-------------+

Update an existing secret's value or auth configuration




.. _mlflowUpdateGatewaySecret:

Request Structure
-----------------






+--------------+--------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
|  Field Name  |                             Type                             |                                                      Description                                                       |
+==============+==============================================================+========================================================================================================================+
| secret_id    | ``STRING``                                                   | ID of the secret to update                                                                                             |
+--------------+--------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| secret_value | An array of :ref:`mlflowupdategatewaysecretsecretvalueentry` | Optional new secret value(s) for key rotation as key-value pairs (empty map = no change).                              |
|              |                                                              | For simple API keys: {"api_key": "sk-xxx"}                                                                             |
|              |                                                              | For compound credentials: {"aws_access_key_id": "...", "aws_secret_access_key": "..."}                                 |
+--------------+--------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| auth_config  | An array of :ref:`mlflowupdategatewaysecretauthconfigentry`  | Optional new auth configuration.                                                                                       |
|              |                                                              | For multi-auth providers, include "auth_mode" key (e.g., {"auth_mode": "access_keys", "aws_region_name": "us-east-1"}) |
+--------------+--------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| updated_by   | ``STRING``                                                   | Username of the updater                                                                                                |
+--------------+--------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+

.. _mlflowUpdateGatewaySecretResponse:

Response Structure
------------------






+------------+--------------------------------+-----------------------------+
| Field Name |              Type              |         Description         |
+============+================================+=============================+
| secret     | :ref:`mlflowgatewaysecretinfo` | The updated secret metadata |
+------------+--------------------------------+-----------------------------+

===========================



.. _mlflowMlflowServicedeleteGatewaySecret:

Delete Gateway Secret
=====================


+---------------------------------------+-------------+
|               Endpoint                | HTTP Method |
+=======================================+=============+
| ``2.0/mlflow/gateway/secrets/delete`` | ``DELETE``  |
+---------------------------------------+-------------+

Delete a secret




.. _mlflowDeleteGatewaySecret:

Request Structure
-----------------






+------------+------------+----------------------------+
| Field Name |    Type    |        Description         |
+============+============+============================+
| secret_id  | ``STRING`` | ID of the secret to delete |
+------------+------------+----------------------------+

===========================



.. _mlflowMlflowServicelistGatewaySecretInfos:

List Gateway Secrets
====================


+-------------------------------------+-------------+
|              Endpoint               | HTTP Method |
+=====================================+=============+
| ``2.0/mlflow/gateway/secrets/list`` | ``GET``     |
+-------------------------------------+-------------+

List all secrets with optional filtering by provider




.. _mlflowListGatewaySecretInfos:

Request Structure
-----------------






+------------+------------+-----------------------------------------------------------+
| Field Name |    Type    |                        Description                        |
+============+============+===========================================================+
| provider   | ``STRING`` | Optional filter by provider (e.g., "openai", "anthropic") |
+------------+------------+-----------------------------------------------------------+

.. _mlflowListGatewaySecretInfosResponse:

Response Structure
------------------






+------------+--------------------------------------------+-------------------------------------------------------------+
| Field Name |                    Type                    |                         Description                         |
+============+============================================+=============================================================+
| secrets    | An array of :ref:`mlflowgatewaysecretinfo` | List of secret metadata (does not include encrypted values) |
+------------+--------------------------------------------+-------------------------------------------------------------+

===========================



.. _mlflowMlflowServicecreateGatewayModelDefinition:

Create Gateway Model Definition
===============================


+-------------------------------------------------+-------------+
|                    Endpoint                     | HTTP Method |
+=================================================+=============+
| ``2.0/mlflow/gateway/model-definitions/create`` | ``POST``    |
+-------------------------------------------------+-------------+

Create a reusable model definition




.. _mlflowCreateGatewayModelDefinition:

Request Structure
-----------------






+------------+------------+--------------------------------------------------------------------------+
| Field Name |    Type    |                               Description                                |
+============+============+==========================================================================+
| name       | ``STRING`` | User-friendly name for the model definition (must be unique)             |
+------------+------------+--------------------------------------------------------------------------+
| secret_id  | ``STRING`` | ID of the secret containing authentication credentials                   |
+------------+------------+--------------------------------------------------------------------------+
| provider   | ``STRING`` | LLM provider (e.g., "openai", "anthropic")                               |
+------------+------------+--------------------------------------------------------------------------+
| model_name | ``STRING`` | Provider-specific model identifier (e.g., "gpt-4o", "claude-3-5-sonnet") |
+------------+------------+--------------------------------------------------------------------------+
| created_by | ``STRING`` | Username of the creator                                                  |
+------------+------------+--------------------------------------------------------------------------+

.. _mlflowCreateGatewayModelDefinitionResponse:

Response Structure
------------------






+------------------+-------------------------------------+------------------------------+
|    Field Name    |                Type                 |         Description          |
+==================+=====================================+==============================+
| model_definition | :ref:`mlflowgatewaymodeldefinition` | The created model definition |
+------------------+-------------------------------------+------------------------------+

===========================



.. _mlflowMlflowServicegetGatewayModelDefinition:

Get Gateway Model Definition
============================


+----------------------------------------------+-------------+
|                   Endpoint                   | HTTP Method |
+==============================================+=============+
| ``2.0/mlflow/gateway/model-definitions/get`` | ``GET``     |
+----------------------------------------------+-------------+

Get a model definition by ID




.. _mlflowGetGatewayModelDefinition:

Request Structure
-----------------






+---------------------+------------+----------------------------------------+
|     Field Name      |    Type    |              Description               |
+=====================+============+========================================+
| model_definition_id | ``STRING`` | ID of the model definition to retrieve |
+---------------------+------------+----------------------------------------+

.. _mlflowGetGatewayModelDefinitionResponse:

Response Structure
------------------






+------------------+-------------------------------------+----------------------+
|    Field Name    |                Type                 |     Description      |
+==================+=====================================+======================+
| model_definition | :ref:`mlflowgatewaymodeldefinition` | The model definition |
+------------------+-------------------------------------+----------------------+

===========================



.. _mlflowMlflowServicelistGatewayModelDefinitions:

List Gateway Model Definitions
==============================


+-----------------------------------------------+-------------+
|                   Endpoint                    | HTTP Method |
+===============================================+=============+
| ``2.0/mlflow/gateway/model-definitions/list`` | ``GET``     |
+-----------------------------------------------+-------------+

List all model definitions with optional filters




.. _mlflowListGatewayModelDefinitions:

Request Structure
-----------------






+------------+------------+------------------------------+
| Field Name |    Type    |         Description          |
+============+============+==============================+
| provider   | ``STRING`` | Optional filter by provider  |
+------------+------------+------------------------------+
| secret_id  | ``STRING`` | Optional filter by secret ID |
+------------+------------+------------------------------+

.. _mlflowListGatewayModelDefinitionsResponse:

Response Structure
------------------






+-------------------+-------------------------------------------------+---------------------------+
|    Field Name     |                      Type                       |        Description        |
+===================+=================================================+===========================+
| model_definitions | An array of :ref:`mlflowgatewaymodeldefinition` | List of model definitions |
+-------------------+-------------------------------------------------+---------------------------+

===========================



.. _mlflowMlflowServiceupdateGatewayModelDefinition:

Update Gateway Model Definition
===============================


+-------------------------------------------------+-------------+
|                    Endpoint                     | HTTP Method |
+=================================================+=============+
| ``2.0/mlflow/gateway/model-definitions/update`` | ``POST``    |
+-------------------------------------------------+-------------+

Update a model definition




.. _mlflowUpdateGatewayModelDefinition:

Request Structure
-----------------






+---------------------+------------+--------------------------------------+
|     Field Name      |    Type    |             Description              |
+=====================+============+======================================+
| model_definition_id | ``STRING`` | ID of the model definition to update |
+---------------------+------------+--------------------------------------+
| name                | ``STRING`` | Optional new name                    |
+---------------------+------------+--------------------------------------+
| secret_id           | ``STRING`` | Optional new secret ID               |
+---------------------+------------+--------------------------------------+
| model_name          | ``STRING`` | Optional new model name              |
+---------------------+------------+--------------------------------------+
| updated_by          | ``STRING`` | Username of the updater              |
+---------------------+------------+--------------------------------------+
| provider            | ``STRING`` | Optional new provider                |
+---------------------+------------+--------------------------------------+

.. _mlflowUpdateGatewayModelDefinitionResponse:

Response Structure
------------------






+------------------+-------------------------------------+------------------------------+
|    Field Name    |                Type                 |         Description          |
+==================+=====================================+==============================+
| model_definition | :ref:`mlflowgatewaymodeldefinition` | The updated model definition |
+------------------+-------------------------------------+------------------------------+

===========================



.. _mlflowMlflowServicedeleteGatewayModelDefinition:

Delete Gateway Model Definition
===============================


+-------------------------------------------------+-------------+
|                    Endpoint                     | HTTP Method |
+=================================================+=============+
| ``2.0/mlflow/gateway/model-definitions/delete`` | ``DELETE``  |
+-------------------------------------------------+-------------+

Delete a model definition (fails if in use by any endpoint)




.. _mlflowDeleteGatewayModelDefinition:

Request Structure
-----------------






+---------------------+------------+------------------------------------------------------------------------+
|     Field Name      |    Type    |                              Description                               |
+=====================+============+========================================================================+
| model_definition_id | ``STRING`` | ID of the model definition to delete (fails if in use by any endpoint) |
+---------------------+------------+------------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicecreateGatewayEndpoint:

Create Gateway Endpoint
=======================


+-----------------------------------------+-------------+
|                Endpoint                 | HTTP Method |
+=========================================+=============+
| ``2.0/mlflow/gateway/endpoints/create`` | ``POST``    |
+-----------------------------------------+-------------+

Create a new endpoint with model configurations




.. _mlflowCreateGatewayEndpoint:

Request Structure
-----------------






+------------------+-----------------------------------------------------+-------------------------------------------------------------------+
|    Field Name    |                        Type                         |                            Description                            |
+==================+=====================================================+===================================================================+
| name             | ``STRING``                                          | Optional user-friendly name for the endpoint                      |
+------------------+-----------------------------------------------------+-------------------------------------------------------------------+
| model_configs    | An array of :ref:`mlflowgatewayendpointmodelconfig` | List of model configurations                                      |
+------------------+-----------------------------------------------------+-------------------------------------------------------------------+
| created_by       | ``STRING``                                          | Username of the creator                                           |
+------------------+-----------------------------------------------------+-------------------------------------------------------------------+
| routing_strategy | :ref:`mlflowroutingstrategy`                        | Optional routing strategy for the endpoint                        |
+------------------+-----------------------------------------------------+-------------------------------------------------------------------+
| fallback_config  | :ref:`mlflowfallbackconfig`                         | Optional fallback configuration (includes strategy, max_attempts) |
+------------------+-----------------------------------------------------+-------------------------------------------------------------------+

.. _mlflowCreateGatewayEndpointResponse:

Response Structure
------------------






+------------+------------------------------+----------------------------------------------+
| Field Name |             Type             |                 Description                  |
+============+==============================+==============================================+
| endpoint   | :ref:`mlflowgatewayendpoint` | The created endpoint with all model mappings |
+------------+------------------------------+----------------------------------------------+

===========================



.. _mlflowMlflowServicegetGatewayEndpoint:

Get Gateway Endpoint
====================


+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/mlflow/gateway/endpoints/get`` | ``GET``     |
+--------------------------------------+-------------+

Get endpoint details including all model configurations




.. _mlflowGetGatewayEndpoint:

Request Structure
-----------------






+-------------+------------+---------------------------------------------+
| Field Name  |    Type    |                 Description                 |
+=============+============+=============================================+
| endpoint_id | ``STRING`` | Either endpoint_id or name must be provided |
+-------------+------------+---------------------------------------------+
| name        | ``STRING`` |                                             |
+-------------+------------+---------------------------------------------+

.. _mlflowGetGatewayEndpointResponse:

Response Structure
------------------






+------------+------------------------------+--------------------------------------------+
| Field Name |             Type             |                Description                 |
+============+==============================+============================================+
| endpoint   | :ref:`mlflowgatewayendpoint` | The endpoint with all model configurations |
+------------+------------------------------+--------------------------------------------+

===========================



.. _mlflowMlflowServiceupdateGatewayEndpoint:

Update Gateway Endpoint
=======================


+-----------------------------------------+-------------+
|                Endpoint                 | HTTP Method |
+=========================================+=============+
| ``2.0/mlflow/gateway/endpoints/update`` | ``POST``    |
+-----------------------------------------+-------------+

Update an endpoint's name




.. _mlflowUpdateGatewayEndpoint:

Request Structure
-----------------






+------------------+-----------------------------------------------------+----------------------------------------------------------------------------------+
|    Field Name    |                        Type                         |                                   Description                                    |
+==================+=====================================================+==================================================================================+
| endpoint_id      | ``STRING``                                          | ID of the endpoint to update                                                     |
+------------------+-----------------------------------------------------+----------------------------------------------------------------------------------+
| name             | ``STRING``                                          | Optional new name for the endpoint                                               |
+------------------+-----------------------------------------------------+----------------------------------------------------------------------------------+
| updated_by       | ``STRING``                                          | Username of the updater                                                          |
+------------------+-----------------------------------------------------+----------------------------------------------------------------------------------+
| model_configs    | An array of :ref:`mlflowgatewayendpointmodelconfig` | Optional new list of model configurations (replaces all existing model linkages) |
+------------------+-----------------------------------------------------+----------------------------------------------------------------------------------+
| routing_strategy | :ref:`mlflowroutingstrategy`                        | Optional new routing strategy for the endpoint                                   |
+------------------+-----------------------------------------------------+----------------------------------------------------------------------------------+
| fallback_config  | :ref:`mlflowfallbackconfig`                         | Optional fallback configuration (includes strategy, max_attempts)                |
+------------------+-----------------------------------------------------+----------------------------------------------------------------------------------+

.. _mlflowUpdateGatewayEndpointResponse:

Response Structure
------------------






+------------+------------------------------+----------------------+
| Field Name |             Type             |     Description      |
+============+==============================+======================+
| endpoint   | :ref:`mlflowgatewayendpoint` | The updated endpoint |
+------------+------------------------------+----------------------+

===========================



.. _mlflowMlflowServicedeleteGatewayEndpoint:

Delete Gateway Endpoint
=======================


+-----------------------------------------+-------------+
|                Endpoint                 | HTTP Method |
+=========================================+=============+
| ``2.0/mlflow/gateway/endpoints/delete`` | ``DELETE``  |
+-----------------------------------------+-------------+

Delete an endpoint and all its model configurations




.. _mlflowDeleteGatewayEndpoint:

Request Structure
-----------------






+-------------+------------+------------------------------+
| Field Name  |    Type    |         Description          |
+=============+============+==============================+
| endpoint_id | ``STRING`` | ID of the endpoint to delete |
+-------------+------------+------------------------------+

===========================



.. _mlflowMlflowServicelistGatewayEndpoints:

List Gateway Endpoints
======================


+---------------------------------------+-------------+
|               Endpoint                | HTTP Method |
+=======================================+=============+
| ``2.0/mlflow/gateway/endpoints/list`` | ``GET``     |
+---------------------------------------+-------------+

List endpoints with optional filtering by provider or secret




.. _mlflowListGatewayEndpoints:

Request Structure
-----------------






+------------+------------+------------------------------+
| Field Name |    Type    |         Description          |
+============+============+==============================+
| provider   | ``STRING`` | Optional filter by provider  |
+------------+------------+------------------------------+
| secret_id  | ``STRING`` | Optional filter by secret ID |
+------------+------------+------------------------------+

.. _mlflowListGatewayEndpointsResponse:

Response Structure
------------------






+------------+------------------------------------------+---------------------------------------------------+
| Field Name |                   Type                   |                    Description                    |
+============+==========================================+===================================================+
| endpoints  | An array of :ref:`mlflowgatewayendpoint` | List of endpoints with their model configurations |
+------------+------------------------------------------+---------------------------------------------------+

===========================



.. _mlflowMlflowServiceattachModelToEndpoint:

Attach Model to Endpoint
========================


+------------------------------------------------+-------------+
|                    Endpoint                    | HTTP Method |
+================================================+=============+
| ``2.0/mlflow/gateway/endpoints/models/attach`` | ``POST``    |
+------------------------------------------------+-------------+

Attach an existing model definition to an endpoint




.. _mlflowAttachModelToGatewayEndpoint:

Request Structure
-----------------






+--------------+-----------------------------------------+-------------------------------------------+
|  Field Name  |                  Type                   |                Description                |
+==============+=========================================+===========================================+
| endpoint_id  | ``STRING``                              | ID of the endpoint to attach the model to |
+--------------+-----------------------------------------+-------------------------------------------+
| model_config | :ref:`mlflowgatewayendpointmodelconfig` | Configuration for the model to attach     |
+--------------+-----------------------------------------+-------------------------------------------+
| created_by   | ``STRING``                              | Username of the creator                   |
+--------------+-----------------------------------------+-------------------------------------------+

.. _mlflowAttachModelToGatewayEndpointResponse:

Response Structure
------------------






+------------+------------------------------------------+---------------------+
| Field Name |                   Type                   |     Description     |
+============+==========================================+=====================+
| mapping    | :ref:`mlflowgatewayendpointmodelmapping` | The created mapping |
+------------+------------------------------------------+---------------------+

===========================



.. _mlflowMlflowServicedetachModelFromEndpoint:

Detach Model from Endpoint
==========================


+------------------------------------------------+-------------+
|                    Endpoint                    | HTTP Method |
+================================================+=============+
| ``2.0/mlflow/gateway/endpoints/models/detach`` | ``POST``    |
+------------------------------------------------+-------------+

Detach a model definition from an endpoint (does not delete the model definition)




.. _mlflowDetachModelFromGatewayEndpoint:

Request Structure
-----------------






+---------------------+------------+--------------------------------------+
|     Field Name      |    Type    |             Description              |
+=====================+============+======================================+
| endpoint_id         | ``STRING`` | ID of the endpoint                   |
+---------------------+------------+--------------------------------------+
| model_definition_id | ``STRING`` | ID of the model definition to detach |
+---------------------+------------+--------------------------------------+

===========================



.. _mlflowMlflowServicecreateEndpointBinding:

Create Endpoint Binding
=======================


+--------------------------------------------------+-------------+
|                     Endpoint                     | HTTP Method |
+==================================================+=============+
| ``2.0/mlflow/gateway/endpoints/bindings/create`` | ``POST``    |
+--------------------------------------------------+-------------+

Create a binding between an endpoint and an MLflow resource




.. _mlflowCreateGatewayEndpointBinding:

Request Structure
-----------------






+---------------+------------+-----------------------------+
|  Field Name   |    Type    |         Description         |
+===============+============+=============================+
| endpoint_id   | ``STRING`` | ID of the endpoint to bind  |
+---------------+------------+-----------------------------+
| resource_type | ``STRING`` | Type of MLflow resource     |
+---------------+------------+-----------------------------+
| resource_id   | ``STRING`` | ID of the resource instance |
+---------------+------------+-----------------------------+
| created_by    | ``STRING`` | Username of the creator     |
+---------------+------------+-----------------------------+

.. _mlflowCreateGatewayEndpointBindingResponse:

Response Structure
------------------






+------------+-------------------------------------+---------------------+
| Field Name |                Type                 |     Description     |
+============+=====================================+=====================+
| binding    | :ref:`mlflowgatewayendpointbinding` | The created binding |
+------------+-------------------------------------+---------------------+

===========================



.. _mlflowMlflowServicedeleteEndpointBinding:

Delete Endpoint Binding
=======================


+--------------------------------------------------+-------------+
|                     Endpoint                     | HTTP Method |
+==================================================+=============+
| ``2.0/mlflow/gateway/endpoints/bindings/delete`` | ``DELETE``  |
+--------------------------------------------------+-------------+

Delete a binding between an endpoint and a resource




.. _mlflowDeleteGatewayEndpointBinding:

Request Structure
-----------------






+---------------+------------+----------------------------------------+
|  Field Name   |    Type    |              Description               |
+===============+============+========================================+
| endpoint_id   | ``STRING`` | ID of the endpoint                     |
+---------------+------------+----------------------------------------+
| resource_type | ``STRING`` | Type of resource bound to the endpoint |
+---------------+------------+----------------------------------------+
| resource_id   | ``STRING`` | ID of the resource                     |
+---------------+------------+----------------------------------------+

===========================



.. _mlflowMlflowServicelistEndpointBindings:

List Endpoint Bindings
======================


+------------------------------------------------+-------------+
|                    Endpoint                    | HTTP Method |
+================================================+=============+
| ``2.0/mlflow/gateway/endpoints/bindings/list`` | ``GET``     |
+------------------------------------------------+-------------+

List all bindings for an endpoint




.. _mlflowListGatewayEndpointBindings:

Request Structure
-----------------






+---------------+------------+---------------------------------------------------------+
|  Field Name   |    Type    |                       Description                       |
+===============+============+=========================================================+
| endpoint_id   | ``STRING`` | ID of the endpoint to list bindings for                 |
+---------------+------------+---------------------------------------------------------+
| resource_type | ``STRING`` | Type of resource to filter bindings by (e.g., "scorer") |
+---------------+------------+---------------------------------------------------------+
| resource_id   | ``STRING`` | ID of the resource to filter bindings by                |
+---------------+------------+---------------------------------------------------------+

.. _mlflowListGatewayEndpointBindingsResponse:

Response Structure
------------------






+------------+-------------------------------------------------+-----------------------------------+
| Field Name |                      Type                       |            Description            |
+============+=================================================+===================================+
| bindings   | An array of :ref:`mlflowgatewayendpointbinding` | List of bindings for the endpoint |
+------------+-------------------------------------------------+-----------------------------------+

===========================



.. _mlflowMlflowServicesetGatewayEndpointTag:

Gateway Set Endpoint Tag
========================


+------------------------------------------+-------------+
|                 Endpoint                 | HTTP Method |
+==========================================+=============+
| ``2.0/mlflow/gateway/endpoints/set-tag`` | ``POST``    |
+------------------------------------------+-------------+

Set a tag on an endpoint




.. _mlflowSetGatewayEndpointTag:

Request Structure
-----------------






+-------------+------------+----------------------------------+
| Field Name  |    Type    |           Description            |
+=============+============+==================================+
| endpoint_id | ``STRING`` | ID of the endpoint to set tag on |
+-------------+------------+----------------------------------+
| key         | ``STRING`` | Tag key to set                   |
+-------------+------------+----------------------------------+
| value       | ``STRING`` | Tag value to set                 |
+-------------+------------+----------------------------------+

===========================



.. _mlflowMlflowServicedeleteGatewayEndpointTag:

Gateway Delete Endpoint Tag
===========================


+---------------------------------------------+-------------+
|                  Endpoint                   | HTTP Method |
+=============================================+=============+
| ``2.0/mlflow/gateway/endpoints/delete-tag`` | ``DELETE``  |
+---------------------------------------------+-------------+

Delete a tag from an endpoint




.. _mlflowDeleteGatewayEndpointTag:

Request Structure
-----------------






+-------------+------------+---------------------------------------+
| Field Name  |    Type    |              Description              |
+=============+============+=======================================+
| endpoint_id | ``STRING`` | ID of the endpoint to delete tag from |
+-------------+------------+---------------------------------------+
| key         | ``STRING`` | Tag key to delete                     |
+-------------+------------+---------------------------------------+

===========================



.. _mlflowMlflowServicecreatePromptOptimizationJob:

Create Prompt Optimization Job
==============================


+-----------------------------------------+-------------+
|                Endpoint                 | HTTP Method |
+=========================================+=============+
| ``2.0/mlflow/prompt-optimization/jobs`` | ``POST``    |
+-----------------------------------------+-------------+

Create a new prompt optimization job.
This endpoint initiates an optimization run with the specified configuration.
The optimization process runs asynchronously and can be monitored via getPromptOptimizationJob.




.. _mlflowCreatePromptOptimizationJob:

Request Structure
-----------------






+-------------------+---------------------------------------------------+----------------------------------------------------------------------+
|    Field Name     |                       Type                        |                             Description                              |
+===================+===================================================+======================================================================+
| experiment_id     | ``STRING``                                        | ID of the MLflow experiment to track the optimization job in.        |
+-------------------+---------------------------------------------------+----------------------------------------------------------------------+
| source_prompt_uri | ``STRING``                                        | URI of the source prompt to optimize (e.g., "prompts:/my-prompt/1"). |
+-------------------+---------------------------------------------------+----------------------------------------------------------------------+
| config            | :ref:`mlflowpromptoptimizationjobconfig`          | Configuration for the optimization job.                              |
+-------------------+---------------------------------------------------+----------------------------------------------------------------------+
| tags              | An array of :ref:`mlflowpromptoptimizationjobtag` | Optional tags for the optimization job.                              |
+-------------------+---------------------------------------------------+----------------------------------------------------------------------+

.. _mlflowCreatePromptOptimizationJobResponse:

Response Structure
------------------






+------------+------------------------------------+-------------------------------+
| Field Name |                Type                |          Description          |
+============+====================================+===============================+
| job        | :ref:`mlflowpromptoptimizationjob` | The created optimization job. |
+------------+------------------------------------+-------------------------------+

===========================



.. _mlflowMlflowServicegetPromptOptimizationJob:

Get Prompt Optimization Job
===========================


+--------------------------------------------------+-------------+
|                     Endpoint                     | HTTP Method |
+==================================================+=============+
| ``2.0/mlflow/prompt-optimization/jobs/{job_id}`` | ``GET``     |
+--------------------------------------------------+-------------+

Get the details and status of a prompt optimization job.
Returns the job configuration, current status, progress statistics,
and the best prompt if the optimization has completed.




.. _mlflowGetPromptOptimizationJob:

Request Structure
-----------------






+------------+------------+-----------------------------------------------------------------+
| Field Name |    Type    |                           Description                           |
+============+============+=================================================================+
| job_id     | ``STRING`` | The unique identifier of the optimization job (same as run_id). |
+------------+------------+-----------------------------------------------------------------+

.. _mlflowGetPromptOptimizationJobResponse:

Response Structure
------------------






+------------+------------------------------------+-------------------------------+
| Field Name |                Type                |          Description          |
+============+====================================+===============================+
| job        | :ref:`mlflowpromptoptimizationjob` | The optimization job details. |
+------------+------------------------------------+-------------------------------+

===========================



.. _mlflowMlflowServicesearchPromptOptimizationJobs:

Search Prompt Optimization Jobs
===============================


+------------------------------------------------+-------------+
|                    Endpoint                    | HTTP Method |
+================================================+=============+
| ``2.0/mlflow/prompt-optimization/jobs/search`` | ``POST``    |
+------------------------------------------------+-------------+

Search for prompt optimization jobs.
Returns a list of optimization jobs matching the specified filters.




.. _mlflowSearchPromptOptimizationJobs:

Request Structure
-----------------






+---------------+------------+-------------------------------------------------------------+
|  Field Name   |    Type    |                         Description                         |
+===============+============+=============================================================+
| experiment_id | ``STRING`` | ID of the MLflow experiment to search optimization jobs in. |
+---------------+------------+-------------------------------------------------------------+

.. _mlflowSearchPromptOptimizationJobsResponse:

Response Structure
------------------






+------------+------------------------------------------------+----------------------------+
| Field Name |                      Type                      |        Description         |
+============+================================================+============================+
| jobs       | An array of :ref:`mlflowpromptoptimizationjob` | List of optimization jobs. |
+------------+------------------------------------------------+----------------------------+

===========================



.. _mlflowMlflowServicecancelPromptOptimizationJob:

Cancel Prompt Optimization Job
==============================


+---------------------------------------------------------+-------------+
|                        Endpoint                         | HTTP Method |
+=========================================================+=============+
| ``2.0/mlflow/prompt-optimization/jobs/{job_id}/cancel`` | ``POST``    |
+---------------------------------------------------------+-------------+

Cancel an in-progress prompt optimization job.
If the job is already completed or cancelled, this operation has no effect.




.. _mlflowCancelPromptOptimizationJob:

Request Structure
-----------------






+------------+------------+----------------------------------------------------------+
| Field Name |    Type    |                       Description                        |
+============+============+==========================================================+
| job_id     | ``STRING`` | The unique identifier of the optimization job to cancel. |
+------------+------------+----------------------------------------------------------+

.. _mlflowCancelPromptOptimizationJobResponse:

Response Structure
------------------






+------------+------------------------------------+---------------------------------+
| Field Name |                Type                |           Description           |
+============+====================================+=================================+
| job        | :ref:`mlflowpromptoptimizationjob` | The cancelled optimization job. |
+------------+------------------------------------+---------------------------------+

===========================



.. _mlflowMlflowServicedeletePromptOptimizationJob:

Delete Prompt Optimization Job
==============================


+--------------------------------------------------+-------------+
|                     Endpoint                     | HTTP Method |
+==================================================+=============+
| ``2.0/mlflow/prompt-optimization/jobs/{job_id}`` | ``DELETE``  |
+--------------------------------------------------+-------------+

Delete a prompt optimization job and its associated data.
This permanently removes the job and all related information.




.. _mlflowDeletePromptOptimizationJob:

Request Structure
-----------------






+------------+------------+----------------------------------------------------------+
| Field Name |    Type    |                       Description                        |
+============+============+==========================================================+
| job_id     | ``STRING`` | The unique identifier of the optimization job to delete. |
+------------+------------+----------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicecreateRegisteredModel:

Create RegisteredModel
======================


+-----------------------------------------+-------------+
|                Endpoint                 | HTTP Method |
+=========================================+=============+
| ``2.0/mlflow/registered-models/create`` | ``POST``    |
+-----------------------------------------+-------------+

Throws ``RESOURCE_ALREADY_EXISTS`` if a registered model with the given name exists.




.. _mlflowCreateRegisteredModel:

Request Structure
-----------------






+-------------------+---------------------------------------------+--------------------------------------------+
|    Field Name     |                    Type                     |                Description                 |
+===================+=============================================+============================================+
| name              | ``STRING``                                  | Register models under this name            |
|                   |                                             | This field is required.                    |
|                   |                                             |                                            |
+-------------------+---------------------------------------------+--------------------------------------------+
| tags              | An array of :ref:`mlflowregisteredmodeltag` | Additional metadata for registered model.  |
+-------------------+---------------------------------------------+--------------------------------------------+
| description       | ``STRING``                                  | Optional description for registered model. |
+-------------------+---------------------------------------------+--------------------------------------------+
| deployment_job_id | ``STRING``                                  | Deployment job id for this model.          |
+-------------------+---------------------------------------------+--------------------------------------------+

.. _mlflowCreateRegisteredModelResponse:

Response Structure
------------------






+------------------+------------------------------+-------------+
|    Field Name    |             Type             | Description |
+==================+==============================+=============+
| registered_model | :ref:`mlflowregisteredmodel` |             |
+------------------+------------------------------+-------------+

===========================



.. _mlflowModelRegistryServicegetRegisteredModel:

Get RegisteredModel
===================


+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/mlflow/registered-models/get`` | ``GET``     |
+--------------------------------------+-------------+






.. _mlflowGetRegisteredModel:

Request Structure
-----------------






+------------+------------+------------------------------------------+
| Field Name |    Type    |               Description                |
+============+============+==========================================+
| name       | ``STRING`` | Registered model unique name identifier. |
|            |            | This field is required.                  |
|            |            |                                          |
+------------+------------+------------------------------------------+

.. _mlflowGetRegisteredModelResponse:

Response Structure
------------------






+------------------+------------------------------+-------------+
|    Field Name    |             Type             | Description |
+==================+==============================+=============+
| registered_model | :ref:`mlflowregisteredmodel` |             |
+------------------+------------------------------+-------------+

===========================



.. _mlflowModelRegistryServicerenameRegisteredModel:

Rename RegisteredModel
======================


+-----------------------------------------+-------------+
|                Endpoint                 | HTTP Method |
+=========================================+=============+
| ``2.0/mlflow/registered-models/rename`` | ``POST``    |
+-----------------------------------------+-------------+






.. _mlflowRenameRegisteredModel:

Request Structure
-----------------






+------------+------------+--------------------------------------------------------------+
| Field Name |    Type    |                         Description                          |
+============+============+==============================================================+
| name       | ``STRING`` | Registered model unique name identifier.                     |
|            |            | This field is required.                                      |
|            |            |                                                              |
+------------+------------+--------------------------------------------------------------+
| new_name   | ``STRING`` | If provided, updates the name for this ``registered_model``. |
+------------+------------+--------------------------------------------------------------+

.. _mlflowRenameRegisteredModelResponse:

Response Structure
------------------






+------------------+------------------------------+-------------+
|    Field Name    |             Type             | Description |
+==================+==============================+=============+
| registered_model | :ref:`mlflowregisteredmodel` |             |
+------------------+------------------------------+-------------+

===========================



.. _mlflowModelRegistryServiceupdateRegisteredModel:

Update RegisteredModel
======================


+-----------------------------------------+-------------+
|                Endpoint                 | HTTP Method |
+=========================================+=============+
| ``2.0/mlflow/registered-models/update`` | ``PATCH``   |
+-----------------------------------------+-------------+






.. _mlflowUpdateRegisteredModel:

Request Structure
-----------------






+-------------------+------------+---------------------------------------------------------------------+
|    Field Name     |    Type    |                             Description                             |
+===================+============+=====================================================================+
| name              | ``STRING`` | Registered model unique name identifier.                            |
|                   |            | This field is required.                                             |
|                   |            |                                                                     |
+-------------------+------------+---------------------------------------------------------------------+
| description       | ``STRING`` | If provided, updates the description for this ``registered_model``. |
+-------------------+------------+---------------------------------------------------------------------+
| deployment_job_id | ``STRING`` | Deployment job id for this model.                                   |
+-------------------+------------+---------------------------------------------------------------------+

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


+-----------------------------------------+-------------+
|                Endpoint                 | HTTP Method |
+=========================================+=============+
| ``2.0/mlflow/registered-models/delete`` | ``DELETE``  |
+-----------------------------------------+-------------+






.. _mlflowDeleteRegisteredModel:

Request Structure
-----------------






+------------+------------+------------------------------------------+
| Field Name |    Type    |               Description                |
+============+============+==========================================+
| name       | ``STRING`` | Registered model unique name identifier. |
|            |            | This field is required.                  |
|            |            |                                          |
+------------+------------+------------------------------------------+

===========================



.. _mlflowModelRegistryServicesearchRegisteredModels:

Search RegisteredModels
=======================


+-----------------------------------------+-------------+
|                Endpoint                 | HTTP Method |
+=========================================+=============+
| ``2.0/mlflow/registered-models/search`` | ``GET``     |
+-----------------------------------------+-------------+






.. _mlflowSearchRegisteredModels:

Request Structure
-----------------






+-------------+------------------------+--------------------------------------------------------------------------------------------+
| Field Name  |          Type          |                                        Description                                         |
+=============+========================+============================================================================================+
| filter      | ``STRING``             | String filter condition, like "name LIKE 'my-model-name'".                                 |
|             |                        | Interpreted in the backend automatically as "name LIKE '%my-model-name%'".                 |
|             |                        | Single boolean condition, with string values wrapped in single quotes.                     |
+-------------+------------------------+--------------------------------------------------------------------------------------------+
| max_results | ``INT64``              | Maximum number of models desired. Default is 100. Max threshold is 1000.                   |
+-------------+------------------------+--------------------------------------------------------------------------------------------+
| order_by    | An array of ``STRING`` | List of columns for ordering search results, which can include model name and last updated |
|             |                        | timestamp with an optional "DESC" or "ASC" annotation, where "ASC" is the default.         |
|             |                        | Tiebreaks are done by model name ASC.                                                      |
+-------------+------------------------+--------------------------------------------------------------------------------------------+
| page_token  | ``STRING``             | Pagination token to go to the next page based on a previous search query.                  |
+-------------+------------------------+--------------------------------------------------------------------------------------------+

.. _mlflowSearchRegisteredModelsResponse:

Response Structure
------------------






+-------------------+------------------------------------------+------------------------------------------------------+
|    Field Name     |                   Type                   |                     Description                      |
+===================+==========================================+======================================================+
| registered_models | An array of :ref:`mlflowregisteredmodel` | Registered Models that match the search criteria.    |
+-------------------+------------------------------------------+------------------------------------------------------+
| next_page_token   | ``STRING``                               | Pagination token to request the next page of models. |
+-------------------+------------------------------------------+------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicegetLatestVersions:

Get Latest ModelVersions
========================


+------------------------------------------------------+-------------+
|                       Endpoint                       | HTTP Method |
+======================================================+=============+
| ``2.0/mlflow/registered-models/get-latest-versions`` | ``POST``    |
+------------------------------------------------------+-------------+






.. _mlflowGetLatestVersions:

Request Structure
-----------------






+------------+------------------------+------------------------------------------+
| Field Name |          Type          |               Description                |
+============+========================+==========================================+
| name       | ``STRING``             | Registered model unique name identifier. |
|            |                        | This field is required.                  |
|            |                        |                                          |
+------------+------------------------+------------------------------------------+
| stages     | An array of ``STRING`` | List of stages.                          |
+------------+------------------------+------------------------------------------+

.. _mlflowGetLatestVersionsResponse:

Response Structure
------------------






+----------------+---------------------------------------+--------------------------------------------------------------------------------------------------+
|   Field Name   |                 Type                  |                                           Description                                            |
+================+=======================================+==================================================================================================+
| model_versions | An array of :ref:`mlflowmodelversion` | Latest version models for each requests stage. Only return models with current ``READY`` status. |
|                |                                       | If no ``stages`` provided, returns the latest version for each stage, including ``"None"``.      |
+----------------+---------------------------------------+--------------------------------------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicecreateModelVersion:

Create ModelVersion
===================


+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/mlflow/model-versions/create`` | ``POST``    |
+--------------------------------------+-------------+






.. _mlflowCreateModelVersion:

Request Structure
-----------------






+-------------+------------------------------------------+------------------------------------------------------------------------------------------------------------+
| Field Name  |                   Type                   |                                                Description                                                 |
+=============+==========================================+============================================================================================================+
| name        | ``STRING``                               | Register model under this name                                                                             |
|             |                                          | This field is required.                                                                                    |
|             |                                          |                                                                                                            |
+-------------+------------------------------------------+------------------------------------------------------------------------------------------------------------+
| source      | ``STRING``                               | URI indicating the location of the model artifacts.                                                        |
|             |                                          | This field is required.                                                                                    |
|             |                                          |                                                                                                            |
+-------------+------------------------------------------+------------------------------------------------------------------------------------------------------------+
| run_id      | ``STRING``                               | MLflow run ID for correlation, if ``source`` was generated by an experiment run in                         |
|             |                                          | MLflow tracking server                                                                                     |
+-------------+------------------------------------------+------------------------------------------------------------------------------------------------------------+
| tags        | An array of :ref:`mlflowmodelversiontag` | Additional metadata for model version.                                                                     |
+-------------+------------------------------------------+------------------------------------------------------------------------------------------------------------+
| run_link    | ``STRING``                               | MLflow run link - this is the exact link of the run that generated this model version,                     |
|             |                                          | potentially hosted at another instance of MLflow.                                                          |
+-------------+------------------------------------------+------------------------------------------------------------------------------------------------------------+
| description | ``STRING``                               | Optional description for model version.                                                                    |
+-------------+------------------------------------------+------------------------------------------------------------------------------------------------------------+
| model_id    | ``STRING``                               | Optional `model_id` for model version that is used to link the registered model to the source logged model |
+-------------+------------------------------------------+------------------------------------------------------------------------------------------------------------+

.. _mlflowCreateModelVersionResponse:

Response Structure
------------------






+---------------+---------------------------+-----------------------------------------------------------------+
|  Field Name   |           Type            |                           Description                           |
+===============+===========================+=================================================================+
| model_version | :ref:`mlflowmodelversion` | Return new version number generated for this model in registry. |
+---------------+---------------------------+-----------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicegetModelVersion:

Get ModelVersion
================


+-----------------------------------+-------------+
|             Endpoint              | HTTP Method |
+===================================+=============+
| ``2.0/mlflow/model-versions/get`` | ``GET``     |
+-----------------------------------+-------------+






.. _mlflowGetModelVersion:

Request Structure
-----------------






+------------+------------+------------------------------+
| Field Name |    Type    |         Description          |
+============+============+==============================+
| name       | ``STRING`` | Name of the registered model |
|            |            | This field is required.      |
|            |            |                              |
+------------+------------+------------------------------+
| version    | ``STRING`` | Model version number         |
|            |            | This field is required.      |
|            |            |                              |
+------------+------------+------------------------------+

.. _mlflowGetModelVersionResponse:

Response Structure
------------------






+---------------+---------------------------+-------------+
|  Field Name   |           Type            | Description |
+===============+===========================+=============+
| model_version | :ref:`mlflowmodelversion` |             |
+---------------+---------------------------+-------------+

===========================



.. _mlflowModelRegistryServiceupdateModelVersion:

Update ModelVersion
===================


+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/mlflow/model-versions/update`` | ``PATCH``   |
+--------------------------------------+-------------+






.. _mlflowUpdateModelVersion:

Request Structure
-----------------






+-------------+------------+---------------------------------------------------------------------+
| Field Name  |    Type    |                             Description                             |
+=============+============+=====================================================================+
| name        | ``STRING`` | Name of the registered model                                        |
|             |            | This field is required.                                             |
|             |            |                                                                     |
+-------------+------------+---------------------------------------------------------------------+
| version     | ``STRING`` | Model version number                                                |
|             |            | This field is required.                                             |
|             |            |                                                                     |
+-------------+------------+---------------------------------------------------------------------+
| description | ``STRING`` | If provided, updates the description for this ``registered_model``. |
+-------------+------------+---------------------------------------------------------------------+

.. _mlflowUpdateModelVersionResponse:

Response Structure
------------------






+---------------+---------------------------+-----------------------------------------------------------------+
|  Field Name   |           Type            |                           Description                           |
+===============+===========================+=================================================================+
| model_version | :ref:`mlflowmodelversion` | Return new version number generated for this model in registry. |
+---------------+---------------------------+-----------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicedeleteModelVersion:

Delete ModelVersion
===================


+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/mlflow/model-versions/delete`` | ``DELETE``  |
+--------------------------------------+-------------+






.. _mlflowDeleteModelVersion:

Request Structure
-----------------






+------------+------------+------------------------------+
| Field Name |    Type    |         Description          |
+============+============+==============================+
| name       | ``STRING`` | Name of the registered model |
|            |            | This field is required.      |
|            |            |                              |
+------------+------------+------------------------------+
| version    | ``STRING`` | Model version number         |
|            |            | This field is required.      |
|            |            |                              |
+------------+------------+------------------------------+

===========================



.. _mlflowModelRegistryServicesearchModelVersions:

Search ModelVersions
====================


+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/mlflow/model-versions/search`` | ``GET``     |
+--------------------------------------+-------------+






.. _mlflowSearchModelVersions:

Request Structure
-----------------






+-------------+------------------------+----------------------------------------------------------------------------------------------+
| Field Name  |          Type          |                                         Description                                          |
+=============+========================+==============================================================================================+
| filter      | ``STRING``             | String filter condition, like "name='my-model-name'". Must be a single boolean condition,    |
|             |                        | with string values wrapped in single quotes.                                                 |
+-------------+------------------------+----------------------------------------------------------------------------------------------+
| max_results | ``INT64``              | Maximum number of models desired. Max threshold is 200K. Backends may choose a lower default |
|             |                        | value and maximum threshold.                                                                 |
+-------------+------------------------+----------------------------------------------------------------------------------------------+
| order_by    | An array of ``STRING`` | List of columns to be ordered by including model name, version, stage with an                |
|             |                        | optional "DESC" or "ASC" annotation, where "ASC" is the default.                             |
|             |                        | Tiebreaks are done by latest stage transition timestamp, followed by name ASC, followed by   |
|             |                        | version DESC.                                                                                |
+-------------+------------------------+----------------------------------------------------------------------------------------------+
| page_token  | ``STRING``             | Pagination token to go to next page based on previous search query.                          |
+-------------+------------------------+----------------------------------------------------------------------------------------------+

.. _mlflowSearchModelVersionsResponse:

Response Structure
------------------






+-----------------+---------------------------------------+----------------------------------------------------------------------------+
|   Field Name    |                 Type                  |                                Description                                 |
+=================+=======================================+============================================================================+
| model_versions  | An array of :ref:`mlflowmodelversion` | Models that match the search criteria                                      |
+-----------------+---------------------------------------+----------------------------------------------------------------------------+
| next_page_token | ``STRING``                            | Pagination token to request next page of models for the same search query. |
+-----------------+---------------------------------------+----------------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicegetModelVersionDownloadUri:

Get Download URI For ModelVersion Artifacts
===========================================


+------------------------------------------------+-------------+
|                    Endpoint                    | HTTP Method |
+================================================+=============+
| ``2.0/mlflow/model-versions/get-download-uri`` | ``GET``     |
+------------------------------------------------+-------------+






.. _mlflowGetModelVersionDownloadUri:

Request Structure
-----------------






+------------+------------+------------------------------+
| Field Name |    Type    |         Description          |
+============+============+==============================+
| name       | ``STRING`` | Name of the registered model |
|            |            | This field is required.      |
|            |            |                              |
+------------+------------+------------------------------+
| version    | ``STRING`` | Model version number         |
|            |            | This field is required.      |
|            |            |                              |
+------------+------------+------------------------------+

.. _mlflowGetModelVersionDownloadUriResponse:

Response Structure
------------------






+--------------+------------+-------------------------------------------------------------------------+
|  Field Name  |    Type    |                               Description                               |
+==============+============+=========================================================================+
| artifact_uri | ``STRING`` | URI corresponding to where artifacts for this model version are stored. |
+--------------+------------+-------------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicetransitionModelVersionStage:

Transition ModelVersion Stage
=============================


+------------------------------------------------+-------------+
|                    Endpoint                    | HTTP Method |
+================================================+=============+
| ``2.0/mlflow/model-versions/transition-stage`` | ``POST``    |
+------------------------------------------------+-------------+






.. _mlflowTransitionModelVersionStage:

Request Structure
-----------------






+---------------------------+------------+-------------------------------------------------------------------------------------------+
|        Field Name         |    Type    |                                        Description                                        |
+===========================+============+===========================================================================================+
| name                      | ``STRING`` | Name of the registered model                                                              |
|                           |            | This field is required.                                                                   |
|                           |            |                                                                                           |
+---------------------------+------------+-------------------------------------------------------------------------------------------+
| version                   | ``STRING`` | Model version number                                                                      |
|                           |            | This field is required.                                                                   |
|                           |            |                                                                                           |
+---------------------------+------------+-------------------------------------------------------------------------------------------+
| stage                     | ``STRING`` | Transition `model_version` to new stage.                                                  |
|                           |            | This field is required.                                                                   |
|                           |            |                                                                                           |
+---------------------------+------------+-------------------------------------------------------------------------------------------+
| archive_existing_versions | ``BOOL``   | When transitioning a model version to a particular stage, this flag dictates whether all  |
|                           |            | existing model versions in that stage should be atomically moved to the "archived" stage. |
|                           |            | This ensures that at-most-one model version exists in the target stage.                   |
|                           |            | This field is *required* when transitioning a model versions's stage                      |
|                           |            | This field is required.                                                                   |
|                           |            |                                                                                           |
+---------------------------+------------+-------------------------------------------------------------------------------------------+

.. _mlflowTransitionModelVersionStageResponse:

Response Structure
------------------






+---------------+---------------------------+-----------------------+
|  Field Name   |           Type            |      Description      |
+===============+===========================+=======================+
| model_version | :ref:`mlflowmodelversion` | Updated model version |
+---------------+---------------------------+-----------------------+

===========================



.. _mlflowModelRegistryServicesetRegisteredModelTag:

Set Registered Model Tag
========================


+------------------------------------------+-------------+
|                 Endpoint                 | HTTP Method |
+==========================================+=============+
| ``2.0/mlflow/registered-models/set-tag`` | ``POST``    |
+------------------------------------------+-------------+






.. _mlflowSetRegisteredModelTag:

Request Structure
-----------------






+------------+------------+----------------------------------------------------------------------------------------------------------+
| Field Name |    Type    |                                               Description                                                |
+============+============+==========================================================================================================+
| name       | ``STRING`` | Unique name of the model.                                                                                |
|            |            | This field is required.                                                                                  |
|            |            |                                                                                                          |
+------------+------------+----------------------------------------------------------------------------------------------------------+
| key        | ``STRING`` | Name of the tag. Maximum size depends on storage backend.                                                |
|            |            | If a tag with this name already exists, its preexisting value will be replaced by the specified `value`. |
|            |            | All storage backends are guaranteed to support key values up to 250 bytes in size.                       |
|            |            | This field is required.                                                                                  |
|            |            |                                                                                                          |
+------------+------------+----------------------------------------------------------------------------------------------------------+
| value      | ``STRING`` | String value of the tag being logged. Maximum size depends on storage backend.                           |
|            |            | This field is required.                                                                                  |
|            |            |                                                                                                          |
+------------+------------+----------------------------------------------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicesetModelVersionTag:

Set Model Version Tag
=====================


+---------------------------------------+-------------+
|               Endpoint                | HTTP Method |
+=======================================+=============+
| ``2.0/mlflow/model-versions/set-tag`` | ``POST``    |
+---------------------------------------+-------------+






.. _mlflowSetModelVersionTag:

Request Structure
-----------------






+------------+------------+----------------------------------------------------------------------------------------------------------+
| Field Name |    Type    |                                               Description                                                |
+============+============+==========================================================================================================+
| name       | ``STRING`` | Unique name of the model.                                                                                |
|            |            | This field is required.                                                                                  |
|            |            |                                                                                                          |
+------------+------------+----------------------------------------------------------------------------------------------------------+
| version    | ``STRING`` | Model version number.                                                                                    |
|            |            | This field is required.                                                                                  |
|            |            |                                                                                                          |
+------------+------------+----------------------------------------------------------------------------------------------------------+
| key        | ``STRING`` | Name of the tag. Maximum size depends on storage backend.                                                |
|            |            | If a tag with this name already exists, its preexisting value will be replaced by the specified `value`. |
|            |            | All storage backends are guaranteed to support key values up to 250 bytes in size.                       |
|            |            | This field is required.                                                                                  |
|            |            |                                                                                                          |
+------------+------------+----------------------------------------------------------------------------------------------------------+
| value      | ``STRING`` | String value of the tag being logged. Maximum size depends on storage backend.                           |
|            |            | This field is required.                                                                                  |
|            |            |                                                                                                          |
+------------+------------+----------------------------------------------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicedeleteRegisteredModelTag:

Delete Registered Model Tag
===========================


+---------------------------------------------+-------------+
|                  Endpoint                   | HTTP Method |
+=============================================+=============+
| ``2.0/mlflow/registered-models/delete-tag`` | ``DELETE``  |
+---------------------------------------------+-------------+






.. _mlflowDeleteRegisteredModelTag:

Request Structure
-----------------






+------------+------------+-------------------------------------------------------------------------------------------------------------------+
| Field Name |    Type    |                                                    Description                                                    |
+============+============+===================================================================================================================+
| name       | ``STRING`` | Name of the registered model that the tag was logged under.                                                       |
|            |            | This field is required.                                                                                           |
|            |            |                                                                                                                   |
+------------+------------+-------------------------------------------------------------------------------------------------------------------+
| key        | ``STRING`` | Name of the tag. The name must be an exact match; wild-card deletion is not supported. Maximum size is 250 bytes. |
|            |            | This field is required.                                                                                           |
|            |            |                                                                                                                   |
+------------+------------+-------------------------------------------------------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicedeleteModelVersionTag:

Delete Model Version Tag
========================


+------------------------------------------+-------------+
|                 Endpoint                 | HTTP Method |
+==========================================+=============+
| ``2.0/mlflow/model-versions/delete-tag`` | ``DELETE``  |
+------------------------------------------+-------------+






.. _mlflowDeleteModelVersionTag:

Request Structure
-----------------






+------------+------------+-------------------------------------------------------------------------------------------------------------------+
| Field Name |    Type    |                                                    Description                                                    |
+============+============+===================================================================================================================+
| name       | ``STRING`` | Name of the registered model that the tag was logged under.                                                       |
|            |            | This field is required.                                                                                           |
|            |            |                                                                                                                   |
+------------+------------+-------------------------------------------------------------------------------------------------------------------+
| version    | ``STRING`` | Model version number that the tag was logged under.                                                               |
|            |            | This field is required.                                                                                           |
|            |            |                                                                                                                   |
+------------+------------+-------------------------------------------------------------------------------------------------------------------+
| key        | ``STRING`` | Name of the tag. The name must be an exact match; wild-card deletion is not supported. Maximum size is 250 bytes. |
|            |            | This field is required.                                                                                           |
|            |            |                                                                                                                   |
+------------+------------+-------------------------------------------------------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicesetRegisteredModelAlias:

Set Registered Model Alias
==========================


+----------------------------------------+-------------+
|                Endpoint                | HTTP Method |
+========================================+=============+
| ``2.0/mlflow/registered-models/alias`` | ``POST``    |
+----------------------------------------+-------------+






.. _mlflowSetRegisteredModelAlias:

Request Structure
-----------------






+------------+------------+---------------------------------------------------------------------------------------------------------------+
| Field Name |    Type    |                                                  Description                                                  |
+============+============+===============================================================================================================+
| name       | ``STRING`` | Name of the registered model.                                                                                 |
|            |            | This field is required.                                                                                       |
|            |            |                                                                                                               |
+------------+------------+---------------------------------------------------------------------------------------------------------------+
| alias      | ``STRING`` | Name of the alias. Maximum size depends on storage backend.                                                   |
|            |            | If an alias with this name already exists, its preexisting value will be replaced by the specified `version`. |
|            |            | All storage backends are guaranteed to support alias name values up to 256 bytes in size.                     |
|            |            | This field is required.                                                                                       |
|            |            |                                                                                                               |
+------------+------------+---------------------------------------------------------------------------------------------------------------+
| version    | ``STRING`` | Model version number.                                                                                         |
|            |            | This field is required.                                                                                       |
|            |            |                                                                                                               |
+------------+------------+---------------------------------------------------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicedeleteRegisteredModelAlias:

Delete Registered Model Alias
=============================


+----------------------------------------+-------------+
|                Endpoint                | HTTP Method |
+========================================+=============+
| ``2.0/mlflow/registered-models/alias`` | ``DELETE``  |
+----------------------------------------+-------------+






.. _mlflowDeleteRegisteredModelAlias:

Request Structure
-----------------






+------------+------------+---------------------------------------------------------------------------------------------------------------------+
| Field Name |    Type    |                                                     Description                                                     |
+============+============+=====================================================================================================================+
| name       | ``STRING`` | Name of the registered model.                                                                                       |
|            |            | This field is required.                                                                                             |
|            |            |                                                                                                                     |
+------------+------------+---------------------------------------------------------------------------------------------------------------------+
| alias      | ``STRING`` | Name of the alias. The name must be an exact match; wild-card deletion is not supported. Maximum size is 256 bytes. |
|            |            | This field is required.                                                                                             |
|            |            |                                                                                                                     |
+------------+------------+---------------------------------------------------------------------------------------------------------------------+

===========================



.. _mlflowModelRegistryServicegetModelVersionByAlias:

Get Model Version by Alias
==========================


+----------------------------------------+-------------+
|                Endpoint                | HTTP Method |
+========================================+=============+
| ``2.0/mlflow/registered-models/alias`` | ``GET``     |
+----------------------------------------+-------------+






.. _mlflowGetModelVersionByAlias:

Request Structure
-----------------






+------------+------------+-----------------------------------------------+
| Field Name |    Type    |                  Description                  |
+============+============+===============================================+
| name       | ``STRING`` | Name of the registered model.                 |
|            |            | This field is required.                       |
|            |            |                                               |
+------------+------------+-----------------------------------------------+
| alias      | ``STRING`` | Name of the alias. Maximum size is 256 bytes. |
|            |            | This field is required.                       |
|            |            |                                               |
+------------+------------+-----------------------------------------------+

.. _mlflowGetModelVersionByAliasResponse:

Response Structure
------------------






+---------------+---------------------------+-------------+
|  Field Name   |           Type            | Description |
+===============+===========================+=============+
| model_version | :ref:`mlflowmodelversion` |             |
+---------------+---------------------------+-------------+

===========================



.. _mlflowWebhookServicecreateWebhook:

Create Webhook
==============


+-------------------------+-------------+
|        Endpoint         | HTTP Method |
+=========================+=============+
| ``2.0/mlflow/webhooks`` | ``POST``    |
+-------------------------+-------------+






.. _mlflowCreateWebhook:

Request Structure
-----------------



Create webhook request


+-------------+---------------------------------------+------------------------------------------------------+
| Field Name  |                 Type                  |                     Description                      |
+=============+=======================================+======================================================+
| name        | ``STRING``                            | Name of the webhook                                  |
|             |                                       | This field is required.                              |
|             |                                       |                                                      |
+-------------+---------------------------------------+------------------------------------------------------+
| description | ``STRING``                            | Optional description for the webhook                 |
+-------------+---------------------------------------+------------------------------------------------------+
| url         | ``STRING``                            | URL to send webhook events to                        |
|             |                                       | This field is required.                              |
|             |                                       |                                                      |
+-------------+---------------------------------------+------------------------------------------------------+
| events      | An array of :ref:`mlflowwebhookevent` | List of events to subscribe to                       |
|             |                                       | This field is required.                              |
|             |                                       |                                                      |
+-------------+---------------------------------------+------------------------------------------------------+
| secret      | ``STRING``                            | Secret key for HMAC signature verification           |
+-------------+---------------------------------------+------------------------------------------------------+
| status      | :ref:`mlflowwebhookstatus`            | Initial status (defaults to ACTIVE if not specified) |
+-------------+---------------------------------------+------------------------------------------------------+

.. _mlflowCreateWebhookResponse:

Response Structure
------------------






+------------+----------------------+-------------+
| Field Name |         Type         | Description |
+============+======================+=============+
| webhook    | :ref:`mlflowwebhook` |             |
+------------+----------------------+-------------+

===========================



.. _mlflowWebhookServicelistWebhooks:

List Webhooks
=============


+-------------------------+-------------+
|        Endpoint         | HTTP Method |
+=========================+=============+
| ``2.0/mlflow/webhooks`` | ``GET``     |
+-------------------------+-------------+






.. _mlflowListWebhooks:

Request Structure
-----------------



List webhooks request


+-------------+------------+----------------------------------------+
| Field Name  |    Type    |              Description               |
+=============+============+========================================+
| max_results | ``INT32``  | Maximum number of webhooks to return   |
+-------------+------------+----------------------------------------+
| page_token  | ``STRING`` | Pagination token from previous request |
+-------------+------------+----------------------------------------+

.. _mlflowListWebhooksResponse:

Response Structure
------------------






+-----------------+----------------------------------+--------------------------------+
|   Field Name    |               Type               |          Description           |
+=================+==================================+================================+
| webhooks        | An array of :ref:`mlflowwebhook` | List of webhooks               |
+-----------------+----------------------------------+--------------------------------+
| next_page_token | ``STRING``                       | Pagination token for next page |
+-----------------+----------------------------------+--------------------------------+

===========================



.. _mlflowWebhookServicegetWebhook:

Get Webhook
===========


+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/mlflow/webhooks/{webhook_id}`` | ``GET``     |
+--------------------------------------+-------------+






.. _mlflowGetWebhook:

Request Structure
-----------------



Get webhook request


+------------+------------+-------------------------------+
| Field Name |    Type    |          Description          |
+============+============+===============================+
| webhook_id | ``STRING`` | ID of the webhook to retrieve |
|            |            | This field is required.       |
|            |            |                               |
+------------+------------+-------------------------------+

.. _mlflowGetWebhookResponse:

Response Structure
------------------






+------------+----------------------+-------------+
| Field Name |         Type         | Description |
+============+======================+=============+
| webhook    | :ref:`mlflowwebhook` |             |
+------------+----------------------+-------------+

===========================



.. _mlflowWebhookServiceupdateWebhook:

Update Webhook
==============


+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/mlflow/webhooks/{webhook_id}`` | ``PATCH``   |
+--------------------------------------+-------------+






.. _mlflowUpdateWebhook:

Request Structure
-----------------



Update webhook request


+-------------+---------------------------------------+------------------------------------+
| Field Name  |                 Type                  |            Description             |
+=============+=======================================+====================================+
| webhook_id  | ``STRING``                            | ID of the webhook to update        |
|             |                                       | This field is required.            |
|             |                                       |                                    |
+-------------+---------------------------------------+------------------------------------+
| name        | ``STRING``                            | New name for the webhook           |
+-------------+---------------------------------------+------------------------------------+
| description | ``STRING``                            | New description for the webhook    |
+-------------+---------------------------------------+------------------------------------+
| url         | ``STRING``                            | New URL for the webhook            |
+-------------+---------------------------------------+------------------------------------+
| events      | An array of :ref:`mlflowwebhookevent` | New list of events to subscribe to |
+-------------+---------------------------------------+------------------------------------+
| secret      | ``STRING``                            | New secret key for HMAC signature  |
+-------------+---------------------------------------+------------------------------------+
| status      | :ref:`mlflowwebhookstatus`            | New status for the webhook         |
+-------------+---------------------------------------+------------------------------------+

.. _mlflowUpdateWebhookResponse:

Response Structure
------------------






+------------+----------------------+-------------+
| Field Name |         Type         | Description |
+============+======================+=============+
| webhook    | :ref:`mlflowwebhook` |             |
+------------+----------------------+-------------+

===========================



.. _mlflowWebhookServicedeleteWebhook:

Delete Webhook
==============


+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/mlflow/webhooks/{webhook_id}`` | ``DELETE``  |
+--------------------------------------+-------------+






.. _mlflowDeleteWebhook:

Request Structure
-----------------



Delete webhook request


+------------+------------+-----------------------------+
| Field Name |    Type    |         Description         |
+============+============+=============================+
| webhook_id | ``STRING`` | ID of the webhook to delete |
|            |            | This field is required.     |
|            |            |                             |
+------------+------------+-----------------------------+

===========================



.. _mlflowWebhookServicetestWebhook:

Test Webhook
============


+-------------------------------------------+-------------+
|                 Endpoint                  | HTTP Method |
+===========================================+=============+
| ``2.0/mlflow/webhooks/{webhook_id}/test`` | ``POST``    |
+-------------------------------------------+-------------+






.. _mlflowTestWebhook:

Request Structure
-----------------



Test webhook request


+------------+---------------------------+---------------------------------------------------------------------------------+
| Field Name |           Type            |                                   Description                                   |
+============+===========================+=================================================================================+
| webhook_id | ``STRING``                | ID of the webhook to test                                                       |
|            |                           | This field is required.                                                         |
|            |                           |                                                                                 |
+------------+---------------------------+---------------------------------------------------------------------------------+
| event      | :ref:`mlflowwebhookevent` | Optional event type to test. If not specified, defaults to the first event type |
|            |                           | in the webhook's subscribed events.                                             |
+------------+---------------------------+---------------------------------------------------------------------------------+

.. _mlflowTestWebhookResponse:

Response Structure
------------------






+------------+--------------------------------+-------------+
| Field Name |              Type              | Description |
+============+================================+=============+
| result     | :ref:`mlflowwebhooktestresult` |             |
+------------+--------------------------------+-------------+

===========================



.. _mlflowartifactsMlflowArtifactsServicedownloadArtifact:

Download Artifact
=================


+---------------------------------------------------------+-------------+
|                        Endpoint                         | HTTP Method |
+=========================================================+=============+
| ``2.0/mlflow-artifacts/artifacts/<path:artifact_path>`` | ``GET``     |
+---------------------------------------------------------+-------------+






===========================



.. _mlflowartifactsMlflowArtifactsServiceuploadArtifact:

Upload Artifact
===============


+---------------------------------------------------------+-------------+
|                        Endpoint                         | HTTP Method |
+=========================================================+=============+
| ``2.0/mlflow-artifacts/artifacts/<path:artifact_path>`` | ``PUT``     |
+---------------------------------------------------------+-------------+






===========================



.. _mlflowartifactsMlflowArtifactsServicelistArtifacts:

List Artifacts
==============


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/mlflow-artifacts/artifacts`` | ``GET``     |
+------------------------------------+-------------+






.. _mlflowartifactsListArtifacts:

Request Structure
-----------------






+------------+------------+-----------------------------------------------------------------------------------------+
| Field Name |    Type    |                                       Description                                       |
+============+============+=========================================================================================+
| path       | ``STRING`` | Filter artifacts matching this path (a relative path from the root artifact directory). |
+------------+------------+-----------------------------------------------------------------------------------------+

.. _mlflowartifactsListArtifactsResponse:

Response Structure
------------------






+------------+--------------------------------------------+-------------------------------------------+
| Field Name |                    Type                    |                Description                |
+============+============================================+===========================================+
| files      | An array of :ref:`mlflowartifactsfileinfo` | File location and metadata for artifacts. |
+------------+--------------------------------------------+-------------------------------------------+

===========================



.. _mlflowartifactsMlflowArtifactsServicedeleteArtifact:

Delete Artifacts
================


+---------------------------------------------------------+-------------+
|                        Endpoint                         | HTTP Method |
+=========================================================+=============+
| ``2.0/mlflow-artifacts/artifacts/<path:artifact_path>`` | ``DELETE``  |
+---------------------------------------------------------+-------------+






===========================



.. _mlflowartifactsMlflowArtifactsServicecreateMultipartUpload:

Create an Artifact Multipart Upload
===================================


+----------------------------------------------------------+-------------+
|                         Endpoint                         | HTTP Method |
+==========================================================+=============+
| ``2.0/mlflow-artifacts/mpu/create/<path:artifact_path>`` | ``POST``    |
+----------------------------------------------------------+-------------+






.. _mlflowartifactsCreateMultipartUpload:

Request Structure
-----------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| path       | ``STRING`` |             |
+------------+------------+-------------+
| num_parts  | ``INT64``  |             |
+------------+------------+-------------+

.. _mlflowartifactsCreateMultipartUploadResponse:

Response Structure
------------------






+-------------+-------------------------------------------------------------+-------------+
| Field Name  |                            Type                             | Description |
+=============+=============================================================+=============+
| upload_id   | ``STRING``                                                  |             |
+-------------+-------------------------------------------------------------+-------------+
| credentials | An array of :ref:`mlflowartifactsmultipartuploadcredential` |             |
+-------------+-------------------------------------------------------------+-------------+

===========================



.. _mlflowartifactsMlflowArtifactsServicecompleteMultipartUpload:

Complete an Artifact Multipart Upload
=====================================


+------------------------------------------------------------+-------------+
|                          Endpoint                          | HTTP Method |
+============================================================+=============+
| ``2.0/mlflow-artifacts/mpu/complete/<path:artifact_path>`` | ``POST``    |
+------------------------------------------------------------+-------------+






.. _mlflowartifactsCompleteMultipartUpload:

Request Structure
-----------------






+------------+-------------------------------------------------------+-------------+
| Field Name |                         Type                          | Description |
+============+=======================================================+=============+
| path       | ``STRING``                                            |             |
+------------+-------------------------------------------------------+-------------+
| upload_id  | ``STRING``                                            |             |
+------------+-------------------------------------------------------+-------------+
| parts      | An array of :ref:`mlflowartifactsmultipartuploadpart` |             |
+------------+-------------------------------------------------------+-------------+

===========================



.. _mlflowartifactsMlflowArtifactsServiceabortMultipartUpload:

Abort an Artifact Multipart Upload
==================================


+---------------------------------------------------------+-------------+
|                        Endpoint                         | HTTP Method |
+=========================================================+=============+
| ``2.0/mlflow-artifacts/mpu/abort/<path:artifact_path>`` | ``POST``    |
+---------------------------------------------------------+-------------+






.. _mlflowartifactsAbortMultipartUpload:

Request Structure
-----------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| path       | ``STRING`` |             |
+------------+------------+-------------+
| upload_id  | ``STRING`` |             |
+------------+------------+-------------+

.. _RESTadd:

Data Structures
===============



.. _mlflowAddDatasetToExperiments:

AddDatasetToExperiments
-----------------------






+----------------+------------------------+----------------------------------------------+
|   Field Name   |          Type          |                 Description                  |
+================+========================+==============================================+
| dataset_id     | ``STRING``             | Dataset ID to add to experiments             |
|                |                        | This field is required.                      |
|                |                        |                                              |
+----------------+------------------------+----------------------------------------------+
| experiment_ids | An array of ``STRING`` | Experiment IDs to associate with the dataset |
+----------------+------------------------+----------------------------------------------+

.. _mlflowassessmentsAssessment:

Assessment
----------



Data and metadata for an assessment of a trace.


+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
|           Field Name            |                                  Type                                   |                                       Description                                        |
+=================================+=========================================================================+==========================================================================================+
| assessment_id                   | ``STRING``                                                              | Unique ID of the assessment.                                                             |
|                                 |                                                                         | NB: This is not marked as required field via "validate_required", because the message is |
|                                 |                                                                         |  used in the context of creating a new assessment, where the ID is not known.            |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| assessment_name                 | ``STRING``                                                              | Name of the assessment. The name must not contain ".".                                   |
|                                 |                                                                         | This field is required.                                                                  |
|                                 |                                                                         |                                                                                          |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| trace_id                        | ``STRING``                                                              | ID of the trace this assessment is associated with.                                      |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| span_id                         | ``STRING``                                                              | ID of the span if the assessment is for a particular span (optional).                    |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| source                          | :ref:`mlflowassessmentsassessmentsource`                                | The source this assessment came from.                                                    |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| create_time                     | :ref:`googleprotobuftimestamp`                                          | The creation time of this assessment.                                                    |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| last_update_time                | :ref:`googleprotobuftimestamp`                                          | The last update time of this assessment.                                                 |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| rationale                       | ``STRING``                                                              | Justification for the assessment.                                                        |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| metadata                        | An array of :ref:`mlflowassessmentsassessmentmetadataentry`             | Additional metadata describing the assessment and store additional information,          |
|                                 |                                                                         | such as the chunk relevance chunk_index. This metadata is required to be JSON-           |
|                                 |                                                                         | serializable.                                                                            |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| overrides                       | ``STRING``                                                              | The ID of the assessment which this assessment overrides.                                |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| valid                           | ``BOOL``                                                                | Whether this assessment is valid (i.e. has not been superseded)                          |
|                                 |                                                                         | defaults to true, and is set to false if a new superseding assessment                    |
|                                 |                                                                         | is created.                                                                              |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| ``feedback`` OR ``expectation`` | :ref:`mlflowassessmentsfeedback` OR :ref:`mlflowassessmentsexpectation` |                                                                                          |
|                                 |                                                                         |                                                                                          |
|                                 |                                                                         | If ``feedback``, the feedback on the trace from this assessment.                         |
|                                 |                                                                         |                                                                                          |
|                                 |                                                                         |                                                                                          |
|                                 |                                                                         |                                                                                          |
|                                 |                                                                         |                                                                                          |
|                                 |                                                                         |                                                                                          |
|                                 |                                                                         | If ``expectation``, a representation of the guidelines and/or expected response from the |
|                                 |                                                                         | agent.                                                                                   |
+---------------------------------+-------------------------------------------------------------------------+------------------------------------------------------------------------------------------+

.. _mlflowassessmentsAssessmentError:

AssessmentError
---------------






+---------------+------------+---------------------------------------------------------------------------------------------+
|  Field Name   |    Type    |                                         Description                                         |
+===============+============+=============================================================================================+
| error_code    | ``STRING`` | Value of an assessment when an error has occurred.                                          |
+---------------+------------+---------------------------------------------------------------------------------------------+
| error_message | ``STRING`` |                                                                                             |
+---------------+------------+---------------------------------------------------------------------------------------------+
| stack_trace   | ``STRING`` | Stack trace of the error. Truncated to 1000 characters to avoid making TraceInfo too large. |
+---------------+------------+---------------------------------------------------------------------------------------------+

.. _mlflowassessmentsAssessmentSource:

AssessmentSource
----------------






+-------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| Field Name  |                        Type                        |                                                Description                                                |
+=============+====================================================+===========================================================================================================+
| source_type | :ref:`mlflowassessmentsassessmentsourcesourcetype` | The type of the source.                                                                                   |
|             |                                                    | This field is required.                                                                                   |
|             |                                                    |                                                                                                           |
+-------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------+
| source_id   | ``STRING``                                         | Identifier for the source.                                                                                |
|             |                                                    | Example: For human -> user name; for LLM judge -> judge source (databricks or custom); for code -> empty. |
|             |                                                    | This field is required.                                                                                   |
|             |                                                    |                                                                                                           |
+-------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------+

.. _mlflowGatewaySecretInfoAuthConfigEntry:

AuthConfigEntry
---------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowCreateGatewaySecretAuthConfigEntry:

AuthConfigEntry
---------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowUpdateGatewaySecretAuthConfigEntry:

AuthConfigEntry
---------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowBatchGetTraces:

BatchGetTraces
--------------






+------------+------------------------+----------------------------------------------+
| Field Name |          Type          |                 Description                  |
+============+========================+==============================================+
| trace_ids  | An array of ``STRING`` | ID of the traces to fetch. Must be provided. |
+------------+------------------------+----------------------------------------------+

.. _mlflowCalculateTraceFilterCorrelation:

CalculateTraceFilterCorrelation
-------------------------------






+----------------+------------------------+-------------------------------------------------------------------------+
|   Field Name   |          Type          |                               Description                               |
+================+========================+=========================================================================+
| experiment_ids | An array of ``STRING`` | List of experiment IDs to search within.                                |
+----------------+------------------------+-------------------------------------------------------------------------+
| filter_string1 | ``STRING``             | First filter condition (e.g., "span.type = 'LLM'").                     |
+----------------+------------------------+-------------------------------------------------------------------------+
| filter_string2 | ``STRING``             | Second filter condition (e.g., "feedback.quality > 0.8").               |
+----------------+------------------------+-------------------------------------------------------------------------+
| base_filter    | ``STRING``             | Optional base filter that both filter1 and filter2 are tested on top of |
|                |                        | (e.g., 'request_time > ... and request_time < ...' for time windows).   |
+----------------+------------------------+-------------------------------------------------------------------------+

.. _mlflowCreateAssessment:

CreateAssessment
----------------






+------------+------------------------------------+---------------------------+
| Field Name |                Type                |        Description        |
+============+====================================+===========================+
| assessment | :ref:`mlflowassessmentsassessment` | The assessment to create. |
|            |                                    | This field is required.   |
|            |                                    |                           |
+------------+------------------------------------+---------------------------+

.. _mlflowCreateDataset:

CreateDataset
-------------






+----------------+----------------------------------------------------+----------------------------------------------------------------------------------------+
|   Field Name   |                        Type                        |                                      Description                                       |
+================+====================================================+========================================================================================+
| name           | ``STRING``                                         | Dataset name                                                                           |
|                |                                                    | This field is required.                                                                |
|                |                                                    |                                                                                        |
+----------------+----------------------------------------------------+----------------------------------------------------------------------------------------+
| experiment_ids | An array of ``STRING``                             | Associated experiment IDs. If not provided, defaults to the current active experiment. |
+----------------+----------------------------------------------------+----------------------------------------------------------------------------------------+
| source_type    | :ref:`mlflowdatasetsdatasetrecordsourcesourcetype` | Source type                                                                            |
+----------------+----------------------------------------------------+----------------------------------------------------------------------------------------+
| source         | ``STRING``                                         | Source information                                                                     |
+----------------+----------------------------------------------------+----------------------------------------------------------------------------------------+
| schema         | ``STRING``                                         | Schema information (JSON)                                                              |
+----------------+----------------------------------------------------+----------------------------------------------------------------------------------------+
| profile        | ``STRING``                                         | Profile information (JSON)                                                             |
+----------------+----------------------------------------------------+----------------------------------------------------------------------------------------+
| created_by     | ``STRING``                                         | User creating the dataset                                                              |
+----------------+----------------------------------------------------+----------------------------------------------------------------------------------------+
| tags           | ``STRING``                                         | Tags to set on the dataset (JSON string mapping keys to values)                        |
+----------------+----------------------------------------------------+----------------------------------------------------------------------------------------+

.. _mlflowCreateLoggedModel:

CreateLoggedModel
-----------------






+---------------+-----------------------------------------------+-------------------------------------------------------------------------------+
|  Field Name   |                     Type                      |                                  Description                                  |
+===============+===============================================+===============================================================================+
| experiment_id | ``STRING``                                    | ID of the associated experiment.                                              |
|               |                                               | This field is required.                                                       |
|               |                                               |                                                                               |
+---------------+-----------------------------------------------+-------------------------------------------------------------------------------+
| name          | ``STRING``                                    | Name of the model. Optional. If not specified, the backend will generate one. |
+---------------+-----------------------------------------------+-------------------------------------------------------------------------------+
| model_type    | ``STRING``                                    | The type of model, such as "Agent", "Classifier", "LLM".                      |
+---------------+-----------------------------------------------+-------------------------------------------------------------------------------+
| source_run_id | ``STRING``                                    | Run ID of the run that created this model.                                    |
+---------------+-----------------------------------------------+-------------------------------------------------------------------------------+
| params        | An array of :ref:`mlflowloggedmodelparameter` | LoggedModel params.                                                           |
+---------------+-----------------------------------------------+-------------------------------------------------------------------------------+
| tags          | An array of :ref:`mlflowloggedmodeltag`       | LoggedModel tags.                                                             |
+---------------+-----------------------------------------------+-------------------------------------------------------------------------------+

.. _mlflowDataset:

Dataset
-------



Dataset. Represents a reference to data used for training, testing, or evaluation during
the model development process.


+-------------+------------+----------------------------------------------------------------------------------------------+
| Field Name  |    Type    |                                         Description                                          |
+=============+============+==============================================================================================+
| name        | ``STRING`` | The name of the dataset. E.g. "my.uc.table@2" "nyc-taxi-dataset", "fantastic-elk-3"          |
|             |            | This field is required.                                                                      |
|             |            |                                                                                              |
+-------------+------------+----------------------------------------------------------------------------------------------+
| digest      | ``STRING`` | Dataset digest, e.g. an md5 hash of the dataset that uniquely identifies it                  |
|             |            | within datasets of the same name.                                                            |
|             |            | This field is required.                                                                      |
|             |            |                                                                                              |
+-------------+------------+----------------------------------------------------------------------------------------------+
| source_type | ``STRING`` | The type of the dataset source, e.g. 'databricks-uc-table', 'DBFS', 'S3', ...                |
|             |            | This field is required.                                                                      |
|             |            |                                                                                              |
+-------------+------------+----------------------------------------------------------------------------------------------+
| source      | ``STRING`` | Source information for the dataset. Note that the source may not exactly reproduce the       |
|             |            | dataset if it was transformed / modified before use with MLflow.                             |
|             |            | This field is required.                                                                      |
|             |            |                                                                                              |
+-------------+------------+----------------------------------------------------------------------------------------------+
| schema      | ``STRING`` | The schema of the dataset. E.g., MLflow ColSpec JSON for a dataframe, MLflow TensorSpec JSON |
|             |            | for an ndarray, or another schema format.                                                    |
+-------------+------------+----------------------------------------------------------------------------------------------+
| profile     | ``STRING`` | The profile of the dataset. Summary statistics for the dataset, such as the number of rows   |
|             |            | in a table, the mean / std / mode of each column in a table, or the number of elements       |
|             |            | in an array.                                                                                 |
+-------------+------------+----------------------------------------------------------------------------------------------+

.. _mlflowSearchLoggedModelsDataset:

Dataset
-------






+----------------+------------+----------------------------+
|   Field Name   |    Type    |        Description         |
+================+============+============================+
| dataset_name   | ``STRING`` | The name of the dataset.   |
|                |            | This field is required.    |
|                |            |                            |
+----------------+------------+----------------------------+
| dataset_digest | ``STRING`` | The digest of the dataset. |
+----------------+------------+----------------------------+

.. _mlflowdatasetsDataset:

Dataset
-------






+------------------+------------------------+----------------------------------------------------------------------+
|    Field Name    |          Type          |                             Description                              |
+==================+========================+======================================================================+
| dataset_id       | ``STRING``             | Unique identifier for the dataset                                    |
+------------------+------------------------+----------------------------------------------------------------------+
| name             | ``STRING``             | Dataset name (user-friendly identifier)                              |
+------------------+------------------------+----------------------------------------------------------------------+
| tags             | ``STRING``             | Tags as JSON string (key-value pairs for metadata)                   |
+------------------+------------------------+----------------------------------------------------------------------+
| schema           | ``STRING``             | Schema information (JSON)                                            |
+------------------+------------------------+----------------------------------------------------------------------+
| profile          | ``STRING``             | Profile information (JSON)                                           |
+------------------+------------------------+----------------------------------------------------------------------+
| digest           | ``STRING``             | Dataset digest for integrity checking                                |
+------------------+------------------------+----------------------------------------------------------------------+
| created_time     | ``INT64``              | Creation timestamp in milliseconds                                   |
+------------------+------------------------+----------------------------------------------------------------------+
| last_update_time | ``INT64``              | Last update timestamp in milliseconds                                |
+------------------+------------------------+----------------------------------------------------------------------+
| created_by       | ``STRING``             | User who created the dataset                                         |
+------------------+------------------------+----------------------------------------------------------------------+
| last_updated_by  | ``STRING``             | User who last updated the dataset                                    |
+------------------+------------------------+----------------------------------------------------------------------+
| experiment_ids   | An array of ``STRING`` | Associated experiment IDs (populated from entity_associations table) |
+------------------+------------------------+----------------------------------------------------------------------+

.. _mlflowDatasetInput:

DatasetInput
------------



DatasetInput. Represents a dataset and input tags.


+------------+-----------------------------------+----------------------------------------------------------------------------------+
| Field Name |               Type                |                                   Description                                    |
+============+===================================+==================================================================================+
| tags       | An array of :ref:`mlflowinputtag` | A list of tags for the dataset input, e.g. a "context" tag with value "training" |
+------------+-----------------------------------+----------------------------------------------------------------------------------+
| dataset    | :ref:`mlflowdataset`              | The dataset being used as a Run input.                                           |
|            |                                   | This field is required.                                                          |
|            |                                   |                                                                                  |
+------------+-----------------------------------+----------------------------------------------------------------------------------+

.. _mlflowdatasetsDatasetRecord:

DatasetRecord
-------------






+-------------------+----------------------------------------------------+----------------------------------------------+
|    Field Name     |                        Type                        |                 Description                  |
+===================+====================================================+==============================================+
| dataset_record_id | ``STRING``                                         | Unique identifier for the record             |
+-------------------+----------------------------------------------------+----------------------------------------------+
| dataset_id        | ``STRING``                                         | ID of the dataset this record belongs to     |
+-------------------+----------------------------------------------------+----------------------------------------------+
| inputs            | ``STRING``                                         | Inputs as JSON string                        |
+-------------------+----------------------------------------------------+----------------------------------------------+
| expectations      | ``STRING``                                         | Expectations as JSON string                  |
+-------------------+----------------------------------------------------+----------------------------------------------+
| tags              | ``STRING``                                         | Tags as JSON string                          |
+-------------------+----------------------------------------------------+----------------------------------------------+
| source            | ``STRING``                                         | Source information as JSON string            |
+-------------------+----------------------------------------------------+----------------------------------------------+
| source_id         | ``STRING``                                         | Source ID for quick lookups (e.g., trace_id) |
+-------------------+----------------------------------------------------+----------------------------------------------+
| source_type       | :ref:`mlflowdatasetsdatasetrecordsourcesourcetype` | Source type                                  |
+-------------------+----------------------------------------------------+----------------------------------------------+
| created_time      | ``INT64``                                          | Creation timestamp in milliseconds           |
+-------------------+----------------------------------------------------+----------------------------------------------+
| last_update_time  | ``INT64``                                          | Last update timestamp in milliseconds        |
+-------------------+----------------------------------------------------+----------------------------------------------+
| created_by        | ``STRING``                                         | User who created the record                  |
+-------------------+----------------------------------------------------+----------------------------------------------+
| last_updated_by   | ``STRING``                                         | User who last updated the record             |
+-------------------+----------------------------------------------------+----------------------------------------------+
| outputs           | ``STRING``                                         | Outputs as JSON string                       |
+-------------------+----------------------------------------------------+----------------------------------------------+

.. _mlflowdatasetsDatasetRecordSource:

DatasetRecordSource
-------------------






+-------------+----------------------------------------------------+------------------------------+
| Field Name  |                        Type                        |         Description          |
+=============+====================================================+==============================+
| source_type | :ref:`mlflowdatasetsdatasetrecordsourcesourcetype` | The type of the source.      |
+-------------+----------------------------------------------------+------------------------------+
| source_data | ``STRING``                                         | Source-specific data as JSON |
+-------------+----------------------------------------------------+------------------------------+

.. _mlflowDatasetSummary:

DatasetSummary
--------------



DatasetSummary. Represents a summary of information about a dataset.


+---------------+------------+-------------------------------------------------------------------------------------+
|  Field Name   |    Type    |                                     Description                                     |
+===============+============+=====================================================================================+
| experiment_id | ``STRING`` | Unique identifier for the experiment.                                               |
|               |            | This field is required.                                                             |
|               |            |                                                                                     |
+---------------+------------+-------------------------------------------------------------------------------------+
| name          | ``STRING`` | The name of the dataset. E.g. "my.uc.table@2" "nyc-taxi-dataset", "fantastic-elk-3" |
|               |            | This field is required.                                                             |
|               |            |                                                                                     |
+---------------+------------+-------------------------------------------------------------------------------------+
| digest        | ``STRING`` | Dataset digest, e.g. an md5 hash of the dataset that uniquely identifies it         |
|               |            | within datasets of the same name.                                                   |
|               |            | This field is required.                                                             |
|               |            |                                                                                     |
+---------------+------------+-------------------------------------------------------------------------------------+
| context       | ``STRING`` | Value of "context" tag if set for the given dataset.                                |
+---------------+------------+-------------------------------------------------------------------------------------+

.. _mlflowDeleteAssessment:

DeleteAssessment
----------------



A request to delete an assessment identified by its trace_id and assessment_id.
The response is empty on successful deletion.


+---------------+------------+---------------------------+
|  Field Name   |    Type    |        Description        |
+===============+============+===========================+
| trace_id      | ``STRING`` | The ID of the trace.      |
|               |            | This field is required.   |
|               |            |                           |
+---------------+------------+---------------------------+
| assessment_id | ``STRING`` | The ID of the assessment. |
|               |            | This field is required.   |
|               |            |                           |
+---------------+------------+---------------------------+

.. _mlflowDeleteDataset:

DeleteDataset
-------------






+------------+------------+-------------------------+
| Field Name |    Type    |       Description       |
+============+============+=========================+
| dataset_id | ``STRING`` | Dataset ID to delete    |
|            |            | This field is required. |
|            |            |                         |
+------------+------------+-------------------------+

.. _mlflowDeleteDatasetTag:

DeleteDatasetTag
----------------






+------------+------------+-------------------------------+
| Field Name |    Type    |          Description          |
+============+============+===============================+
| dataset_id | ``STRING`` | Dataset ID to delete tag from |
|            |            | This field is required.       |
|            |            |                               |
+------------+------------+-------------------------------+
| key        | ``STRING`` | Tag key to delete             |
|            |            | This field is required.       |
|            |            |                               |
+------------+------------+-------------------------------+

.. _mlflowDeleteLoggedModel:

DeleteLoggedModel
-----------------






+------------+------------+--------------------------------------+
| Field Name |    Type    |             Description              |
+============+============+======================================+
| model_id   | ``STRING`` | The ID of the LoggedModel to delete. |
|            |            | This field is required.              |
|            |            |                                      |
+------------+------------+--------------------------------------+

.. _mlflowDeleteLoggedModelTag:

DeleteLoggedModelTag
--------------------






+------------+------------+---------------------------------------------------+
| Field Name |    Type    |                    Description                    |
+============+============+===================================================+
| model_id   | ``STRING`` | The ID of the LoggedModel to delete the tag from. |
|            |            | This field is required.                           |
|            |            |                                                   |
+------------+------------+---------------------------------------------------+
| tag_key    | ``STRING`` | The tag key.                                      |
|            |            | This field is required.                           |
|            |            |                                                   |
+------------+------------+---------------------------------------------------+

.. _mlflowDeleteTraceTag:

DeleteTraceTag
--------------






+------------+------------+-----------------------------------------------+
| Field Name |    Type    |                  Description                  |
+============+============+===============================================+
| request_id | ``STRING`` | ID of the trace from which to delete the tag. |
+------------+------------+-----------------------------------------------+
| key        | ``STRING`` | Name of the tag to delete.                    |
+------------+------------+-----------------------------------------------+

.. _mlflowDeleteTraceTagV3:

DeleteTraceTagV3
----------------






+------------+------------+-----------------------------------------------+
| Field Name |    Type    |                  Description                  |
+============+============+===============================================+
| trace_id   | ``STRING`` | ID of the trace from which to delete the tag. |
+------------+------------+-----------------------------------------------+
| key        | ``STRING`` | Name of the tag to delete.                    |
+------------+------------+-----------------------------------------------+

.. _mlflowDeleteTraces:

DeleteTraces
------------






+----------------------+------------------------+---------------------------------------------------------------------------------------+
|      Field Name      |          Type          |                                      Description                                      |
+======================+========================+=======================================================================================+
| experiment_id        | ``STRING``             | ID of the associated experiment.                                                      |
|                      |                        | This field is required.                                                               |
|                      |                        |                                                                                       |
+----------------------+------------------------+---------------------------------------------------------------------------------------+
| max_timestamp_millis | ``INT64``              | Case 1: max_timestamp_millis and max_traces must be specified for time-based deletion |
|                      |                        | The maximum timestamp in milliseconds since the UNIX epoch for deleting traces.       |
+----------------------+------------------------+---------------------------------------------------------------------------------------+
| max_traces           | ``INT32``              | The maximum number of traces to delete.                                               |
+----------------------+------------------------+---------------------------------------------------------------------------------------+
| request_ids          | An array of ``STRING`` | Case 2: request_ids must be specified for ID-based deletion                           |
|                      |                        | A set of request IDs to delete                                                        |
+----------------------+------------------------+---------------------------------------------------------------------------------------+

.. _mlflowDeleteTracesV3:

DeleteTracesV3
--------------






+----------------------+------------------------+---------------------------------------------------------------------------------------+
|      Field Name      |          Type          |                                      Description                                      |
+======================+========================+=======================================================================================+
| experiment_id        | ``STRING``             | ID of the associated experiment.                                                      |
|                      |                        | This field is required.                                                               |
|                      |                        |                                                                                       |
+----------------------+------------------------+---------------------------------------------------------------------------------------+
| max_timestamp_millis | ``INT64``              | Case 1: max_timestamp_millis and max_traces must be specified for time-based deletion |
|                      |                        | The maximum timestamp in milliseconds since the UNIX epoch for deleting traces.       |
+----------------------+------------------------+---------------------------------------------------------------------------------------+
| max_traces           | ``INT32``              | The maximum number of traces to delete.                                               |
+----------------------+------------------------+---------------------------------------------------------------------------------------+
| request_ids          | An array of ``STRING`` | Case 2: request_ids must be specified for ID-based deletion                           |
|                      |                        | A set of request IDs to delete                                                        |
+----------------------+------------------------+---------------------------------------------------------------------------------------+

.. _mlflowMetricDataPointDimensionsEntry:

DimensionsEntry
---------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowEndTrace:

EndTrace
--------






+------------------+-----------------------------------------------+----------------------------------------------------------------+
|    Field Name    |                     Type                      |                          Description                           |
+==================+===============================================+================================================================+
| request_id       | ``STRING``                                    | ID of the trace to end.                                        |
+------------------+-----------------------------------------------+----------------------------------------------------------------+
| timestamp_ms     | ``INT64``                                     | Unix timestamp of when the trace ended in milliseconds.        |
+------------------+-----------------------------------------------+----------------------------------------------------------------+
| status           | :ref:`mlflowtracestatus`                      | Overall status of the operation being traced (OK, error, etc). |
+------------------+-----------------------------------------------+----------------------------------------------------------------+
| request_metadata | An array of :ref:`mlflowtracerequestmetadata` | Additional metadata about the operation being traced.          |
+------------------+-----------------------------------------------+----------------------------------------------------------------+
| tags             | An array of :ref:`mlflowtracetag`             | Additional tags to add to the trace.                           |
+------------------+-----------------------------------------------+----------------------------------------------------------------+

.. _mlflowassessmentsExpectation:

Expectation
-----------



An expectation for the values or guidelines for the outputs that a model or agent should produce
from the inputs contained in the trace.


+------------------+----------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
|    Field Name    |                        Type                        |                                                     Description                                                     |
+==================+====================================================+=====================================================================================================================+
| value            | :ref:`googleprotobufvalue`                         | The value of the expectation-based assessment. This uses ``google.protobuf.Value`` under the hood to support a      |
|                  |                                                    | flexible schema of expectation values but is validated to constrain it to specific types. This means the value must |
|                  |                                                    | be JSON conforming to one of the following supported types:                                                         |
|                  |                                                    | * Numeric values like integers or floats                                                                            |
|                  |                                                    | * Boolean values                                                                                                    |
|                  |                                                    | * Text value (can contain JSON text the user wishes to store, but it will only be searchable as text)               |
|                  |                                                    | * List values containing only strings (empty lists allowed).                                                        |
|                  |                                                    |                                                                                                                     |
|                  |                                                    | Other values like null, structs, non-string lists etc. will be rejected. However, they can instead be serialized as |
|                  |                                                    | a string and stored in the ``serialized_value`` field instead. Only one of either ``serialized_value`` or ``value`` |
|                  |                                                    | may be defined. We do not support these other formats directly despite using google.protobuf.Value due to security  |
|                  |                                                    | risks around their serialization and deserialization.                                                               |
+------------------+----------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| serialized_value | :ref:`mlflowassessmentsexpectationserializedvalue` | The value of the expecation-based assessment serialized as a string in a specified format. Only one of either       |
|                  |                                                    | ``serialized_value`` or ``value`` may be defined.                                                                   |
+------------------+----------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+

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

.. _mlflowFallbackConfig:

FallbackConfig
--------------



Configuration for fallback routing


+--------------+-------------------------------+-------------------------------------------------------------------------------+
|  Field Name  |             Type              |                                  Description                                  |
+==============+===============================+===============================================================================+
| strategy     | :ref:`mlflowfallbackstrategy` | The fallback strategy.                                                        |
+--------------+-------------------------------+-------------------------------------------------------------------------------+
| max_attempts | ``INT32``                     | The max attempts for fallback routing (cannot exceed number of destinations). |
+--------------+-------------------------------+-------------------------------------------------------------------------------+

.. _mlflowassessmentsFeedback:

Feedback
--------



Feedback provided on the model / agent output(s) contained in the trace


+------------+-----------------------------------------+-------------------------------------------------------------------------------------------------------+
| Field Name |                  Type                   |                                              Description                                              |
+============+=========================================+=======================================================================================================+
| value      | :ref:`googleprotobufvalue`              | Value of the feedback-based assessment.                                                               |
|            |                                         | We use google.protobuf.Value to support a flexible schema of feedback values.                         |
|            |                                         | Supported initial types:                                                                              |
|            |                                         | - Numeric values like integers or floats                                                              |
|            |                                         | - Boolean values                                                                                      |
|            |                                         | - Text value (can contain json text the user wishes to store, but it will only be searchable as text) |
|            |                                         | - Non-empty list values containing only strings                                                       |
|            |                                         | - Other values like structs, non-string lists etc. will be rejected for now                           |
+------------+-----------------------------------------+-------------------------------------------------------------------------------------------------------+
| error      | :ref:`mlflowassessmentsassessmenterror` | An error encountered while generating the feedback. Required if value is set to null.                 |
+------------+-----------------------------------------+-------------------------------------------------------------------------------------------------------+

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

.. _mlflowartifactsFileInfo:

FileInfo
--------






+------------+------------+---------------------------------------------------+
| Field Name |    Type    |                    Description                    |
+============+============+===================================================+
| path       | ``STRING`` | Path relative to the root artifact directory run. |
+------------+------------+---------------------------------------------------+
| is_dir     | ``BOOL``   | Whether the path is a directory.                  |
+------------+------------+---------------------------------------------------+
| file_size  | ``INT64``  | Size in bytes. Unset for directories.             |
+------------+------------+---------------------------------------------------+

.. _mlflowPromptOptimizationJobFinalEvalScoresEntry:

FinalEvalScoresEntry
--------------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``DOUBLE`` |             |
+------------+------------+-------------+

.. _mlflowFinalizeLoggedModel:

FinalizeLoggedModel
-------------------






+------------+--------------------------------+------------------------------------------------------------------------------------+
| Field Name |              Type              |                                    Description                                     |
+============+================================+====================================================================================+
| model_id   | ``STRING``                     | The ID of the LoggedModel to finalize                                              |
|            |                                | This field is required.                                                            |
|            |                                |                                                                                    |
+------------+--------------------------------+------------------------------------------------------------------------------------+
| status     | :ref:`mlflowloggedmodelstatus` | Whether or not the model is ready for use.                                         |
|            |                                | Valid values in this message: ENUM<LOGGED_MODEL_READY, LOGGED_MODEL_UPLOAD_FAILED> |
|            |                                | ("LOGGED_MODEL_UPLOAD_FAILED" indicates that something went wrong when logging     |
|            |                                | the model weights / agent code)                                                    |
|            |                                | This field is required.                                                            |
|            |                                |                                                                                    |
+------------+--------------------------------+------------------------------------------------------------------------------------+

.. _mlflowGatewayEndpoint:

GatewayEndpoint
---------------



Endpoint entity representing an LLM gateway endpoint


+------------------+------------------------------------------------------+-------------------------------------------------------------------------+
|    Field Name    |                         Type                         |                               Description                               |
+==================+======================================================+=========================================================================+
| endpoint_id      | ``STRING``                                           | Unique identifier for the endpoint                                      |
+------------------+------------------------------------------------------+-------------------------------------------------------------------------+
| name             | ``STRING``                                           | User-friendly name for the endpoint                                     |
+------------------+------------------------------------------------------+-------------------------------------------------------------------------+
| created_at       | ``INT64``                                            | Timestamp (milliseconds since epoch) when the endpoint was created      |
+------------------+------------------------------------------------------+-------------------------------------------------------------------------+
| last_updated_at  | ``INT64``                                            | Timestamp (milliseconds since epoch) when the endpoint was last updated |
+------------------+------------------------------------------------------+-------------------------------------------------------------------------+
| model_mappings   | An array of :ref:`mlflowgatewayendpointmodelmapping` | List of model mappings bound to this endpoint                           |
+------------------+------------------------------------------------------+-------------------------------------------------------------------------+
| created_by       | ``STRING``                                           | User ID who created the endpoint                                        |
+------------------+------------------------------------------------------+-------------------------------------------------------------------------+
| last_updated_by  | ``STRING``                                           | User ID who last updated the endpoint                                   |
+------------------+------------------------------------------------------+-------------------------------------------------------------------------+
| tags             | An array of :ref:`mlflowgatewayendpointtag`          | Tags associated with the endpoint                                       |
+------------------+------------------------------------------------------+-------------------------------------------------------------------------+
| routing_strategy | :ref:`mlflowroutingstrategy`                         | Routing strategy for the endpoint                                       |
+------------------+------------------------------------------------------+-------------------------------------------------------------------------+
| fallback_config  | :ref:`mlflowfallbackconfig`                          | Fallback configuration (populated if routing_strategy is FALLBACK)      |
+------------------+------------------------------------------------------+-------------------------------------------------------------------------+

.. _mlflowGatewayEndpointBinding:

GatewayEndpointBinding
----------------------



Binding between an endpoint and an MLflow resource.
Uses composite key (endpoint_id, resource_type, resource_id) for identification.


+-----------------+------------+-----------------------------------------------------------------------------------+
|   Field Name    |    Type    |                                    Description                                    |
+=================+============+===================================================================================+
| endpoint_id     | ``STRING`` | ID of the endpoint this binding references                                        |
+-----------------+------------+-----------------------------------------------------------------------------------+
| resource_type   | ``STRING`` | Type of MLflow resource (e.g., "scorer")                                          |
+-----------------+------------+-----------------------------------------------------------------------------------+
| resource_id     | ``STRING`` | ID of the specific resource instance                                              |
+-----------------+------------+-----------------------------------------------------------------------------------+
| created_at      | ``INT64``  | Timestamp (milliseconds since epoch) when the binding was created                 |
+-----------------+------------+-----------------------------------------------------------------------------------+
| last_updated_at | ``INT64``  | Timestamp (milliseconds since epoch) when the binding was last updated            |
+-----------------+------------+-----------------------------------------------------------------------------------+
| created_by      | ``STRING`` | User ID who created the binding                                                   |
+-----------------+------------+-----------------------------------------------------------------------------------+
| last_updated_by | ``STRING`` | User ID who last updated the binding                                              |
+-----------------+------------+-----------------------------------------------------------------------------------+
| display_name    | ``STRING`` | Fields 8-9 reserved - endpoint_name and model_mappings removed (join client-side) |
|                 |            | Human-readable display name for the resource (e.g., scorer name)                  |
+-----------------+------------+-----------------------------------------------------------------------------------+

.. _mlflowGatewayEndpointModelConfig:

GatewayEndpointModelConfig
--------------------------



Configuration for a model attached to an endpoint


+---------------------+--------------------------------------+----------------------------------------------------------------------------+
|     Field Name      |                 Type                 |                                Description                                 |
+=====================+======================================+============================================================================+
| model_definition_id | ``STRING``                           | ID of the model definition                                                 |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+
| linkage_type        | :ref:`mlflowgatewaymodellinkagetype` | Type of linkage                                                            |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+
| weight              | ``FLOAT``                            | Routing weight for traffic distribution                                    |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+
| fallback_order      | ``INT32``                            | Order for fallback attempts (only for FALLBACK linkages, NULL for PRIMARY) |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+

.. _mlflowGatewayEndpointModelMapping:

GatewayEndpointModelMapping
---------------------------



Mapping between an endpoint and a model definition


+---------------------+--------------------------------------+----------------------------------------------------------------------------+
|     Field Name      |                 Type                 |                                Description                                 |
+=====================+======================================+============================================================================+
| mapping_id          | ``STRING``                           | Unique identifier for this mapping                                         |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+
| endpoint_id         | ``STRING``                           | ID of the endpoint                                                         |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+
| model_definition_id | ``STRING``                           | ID of the model definition                                                 |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+
| model_definition    | :ref:`mlflowgatewaymodeldefinition`  | The full model definition (populated via JOIN)                             |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+
| weight              | ``FLOAT``                            | Routing weight for traffic distribution                                    |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+
| created_at          | ``INT64``                            | Timestamp (milliseconds since epoch) when the mapping was created          |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+
| created_by          | ``STRING``                           | User ID who created the mapping                                            |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+
| linkage_type        | :ref:`mlflowgatewaymodellinkagetype` | Type of linkage                                                            |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+
| fallback_order      | ``INT32``                            | Order for fallback attempts (only for FALLBACK linkages, NULL for PRIMARY) |
+---------------------+--------------------------------------+----------------------------------------------------------------------------+

.. _mlflowGatewayEndpointTag:

GatewayEndpointTag
------------------



Tag associated with an endpoint


+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` | Tag key     |
+------------+------------+-------------+
| value      | ``STRING`` | Tag value   |
+------------+------------+-------------+

.. _mlflowGatewayModelDefinition:

GatewayModelDefinition
----------------------



Reusable model definition that can be shared across endpoints


+---------------------+------------+---------------------------------------------------------------------------------+
|     Field Name      |    Type    |                                   Description                                   |
+=====================+============+=================================================================================+
| model_definition_id | ``STRING`` | Unique identifier for this model definition                                     |
+---------------------+------------+---------------------------------------------------------------------------------+
| name                | ``STRING`` | User-friendly name for identification and reuse                                 |
+---------------------+------------+---------------------------------------------------------------------------------+
| secret_id           | ``STRING`` | ID of the secret containing authentication credentials                          |
+---------------------+------------+---------------------------------------------------------------------------------+
| secret_name         | ``STRING`` | Name of the secret for display purposes                                         |
+---------------------+------------+---------------------------------------------------------------------------------+
| provider            | ``STRING`` | LLM provider (e.g., "openai", "anthropic", "cohere", "bedrock")                 |
+---------------------+------------+---------------------------------------------------------------------------------+
| model_name          | ``STRING`` | Provider-specific model identifier (e.g., "gpt-4o", "claude-3-5-sonnet")        |
+---------------------+------------+---------------------------------------------------------------------------------+
| created_at          | ``INT64``  | Timestamp (milliseconds since epoch) when the model definition was created      |
+---------------------+------------+---------------------------------------------------------------------------------+
| last_updated_at     | ``INT64``  | Timestamp (milliseconds since epoch) when the model definition was last updated |
+---------------------+------------+---------------------------------------------------------------------------------+
| created_by          | ``STRING`` | User ID who created the model definition                                        |
+---------------------+------------+---------------------------------------------------------------------------------+
| last_updated_by     | ``STRING`` | User ID who last updated the model definition                                   |
+---------------------+------------+---------------------------------------------------------------------------------+

.. _mlflowGatewaySecretInfo:

GatewaySecretInfo
-----------------



Secret metadata entity (does not include the decrypted secret value)


+-----------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
|   Field Name    |                            Type                             |                                            Description                                            |
+=================+=============================================================+===================================================================================================+
| secret_id       | ``STRING``                                                  | Unique identifier for the secret (UUID)                                                           |
+-----------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| secret_name     | ``STRING``                                                  | User-friendly name for the secret (must be unique)                                                |
+-----------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| masked_values   | An array of :ref:`mlflowgatewaysecretinfomaskedvaluesentry` | Masked version of the secret values for display as key-value pairs.                               |
|                 |                                                             | For simple API keys: {"api_key": "sk-...xyz123"}                                                  |
|                 |                                                             | For compound credentials: ``{"aws_access_key_id": "AKI...1234", "aws_secret_access_key": "***"}`` |
+-----------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| created_at      | ``INT64``                                                   | Timestamp (milliseconds since epoch) when the secret was created                                  |
+-----------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| last_updated_at | ``INT64``                                                   | Timestamp (milliseconds since epoch) when the secret was last updated                             |
+-----------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| provider        | ``STRING``                                                  | LLM provider identifier (e.g., "openai", "anthropic", "cohere")                                   |
+-----------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| created_by      | ``STRING``                                                  | User ID who created the secret                                                                    |
+-----------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| last_updated_by | ``STRING``                                                  | User ID who last updated the secret                                                               |
+-----------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| auth_config     | An array of :ref:`mlflowgatewaysecretinfoauthconfigentry`   | Provider-specific auth configuration (e.g., auth_mode, region, project_id)                        |
+-----------------+-------------------------------------------------------------+---------------------------------------------------------------------------------------------------+

.. _mlflowGetAssessmentRequest:

GetAssessmentRequest
--------------------






+---------------+------------+------------------------------------------------+
|  Field Name   |    Type    |                  Description                   |
+===============+============+================================================+
| trace_id      | ``STRING`` | The ID of the trace the assessment belongs to. |
|               |            | This field is required.                        |
|               |            |                                                |
+---------------+------------+------------------------------------------------+
| assessment_id | ``STRING`` | The ID of the assessment.                      |
|               |            | This field is required.                        |
|               |            |                                                |
+---------------+------------+------------------------------------------------+

.. _mlflowGetDataset:

GetDataset
----------






+------------+------------+--------------------------------------------+
| Field Name |    Type    |                Description                 |
+============+============+============================================+
| dataset_id | ``STRING`` | Dataset ID                                 |
|            |            | This field is required.                    |
|            |            |                                            |
+------------+------------+--------------------------------------------+
| page_token | ``STRING`` | Optional page token for paginating records |
+------------+------------+--------------------------------------------+

.. _mlflowGetDatasetExperimentIds:

GetDatasetExperimentIds
-----------------------






+------------+------------+--------------------------------------+
| Field Name |    Type    |             Description              |
+============+============+======================================+
| dataset_id | ``STRING`` | Dataset ID to get experiment IDs for |
|            |            | This field is required.              |
|            |            |                                      |
+------------+------------+--------------------------------------+

.. _mlflowGetDatasetRecords:

GetDatasetRecords
-----------------






+-------------+------------+-----------------------------------------------------------+
| Field Name  |    Type    |                        Description                        |
+=============+============+===========================================================+
| dataset_id  | ``STRING`` | Dataset ID to get records for                             |
|             |            | This field is required.                                   |
|             |            |                                                           |
+-------------+------------+-----------------------------------------------------------+
| max_results | ``INT32``  | Optional pagination - maximum number of records to return |
+-------------+------------+-----------------------------------------------------------+
| page_token  | ``STRING`` | Optional pagination token for getting next page           |
+-------------+------------+-----------------------------------------------------------+

.. _mlflowGetLoggedModel:

GetLoggedModel
--------------






+------------+------------+----------------------------------------+
| Field Name |    Type    |              Description               |
+============+============+========================================+
| model_id   | ``STRING`` | The ID of the LoggedModel to retrieve. |
|            |            | This field is required.                |
|            |            |                                        |
+------------+------------+----------------------------------------+

.. _mlflowGetMetricHistoryBulkInterval:

GetMetricHistoryBulkInterval
----------------------------






+-------------+------------------------+---------------------------------------------------------------------------------------------+
| Field Name  |          Type          |                                         Description                                         |
+=============+========================+=============================================================================================+
| run_ids     | An array of ``STRING`` | ID(s) of the run(s) from which to fetch metric values. Must be provided.                    |
+-------------+------------------------+---------------------------------------------------------------------------------------------+
| metric_key  | ``STRING``             | Name of the metric.                                                                         |
|             |                        | This field is required.                                                                     |
|             |                        |                                                                                             |
+-------------+------------------------+---------------------------------------------------------------------------------------------+
| start_step  | ``INT32``              | Optional start step to only fetch metrics after the specified step. Must be defined if      |
|             |                        | end_step is defined.                                                                        |
+-------------+------------------------+---------------------------------------------------------------------------------------------+
| end_step    | ``INT32``              | Optional end step to only fetch metrics before the specified step. Must be defined if       |
|             |                        | start_step is defined.                                                                      |
+-------------+------------------------+---------------------------------------------------------------------------------------------+
| max_results | ``INT32``              | Maximum number of results to fetch per run specified. Must be set to a positive number.     |
|             |                        | Note, in reality, the API returns at most (max_results + # of run IDs) x (# run IDs) metric |
|             |                        | data points.                                                                                |
+-------------+------------------------+---------------------------------------------------------------------------------------------+

.. _mlflowGetOnlineTraceDetails:

GetOnlineTraceDetails
---------------------






+------------------------------+------------+-----------------------------------------------+
|          Field Name          |    Type    |                  Description                  |
+==============================+============+===============================================+
| trace_id                     | ``STRING`` | Trace ID to retrieve                          |
|                              |            | This field is required.                       |
|                              |            |                                               |
+------------------------------+------------+-----------------------------------------------+
| sql_warehouse_id             | ``STRING`` | SQL warehouse to use for query                |
|                              |            | This field is required.                       |
|                              |            |                                               |
+------------------------------+------------+-----------------------------------------------+
| source_inference_table       | ``STRING`` | Source inference table to use for query       |
|                              |            | ie. "ml.bbqiu.codegen_payload"                |
|                              |            | This field is required.                       |
|                              |            |                                               |
+------------------------------+------------+-----------------------------------------------+
| source_databricks_request_id | ``STRING`` | Source databricks request id to use for query |
|                              |            | ie. "8d1992ce-ba3d-49e9-9701-e9b323c5cc8c"    |
|                              |            | This field is required.                       |
|                              |            |                                               |
+------------------------------+------------+-----------------------------------------------+

.. _mlflowGetTrace:

GetTrace
--------






+---------------+------------+----------------------------------------------------+
|  Field Name   |    Type    |                    Description                     |
+===============+============+====================================================+
| trace_id      | ``STRING`` | ID of the trace to fetch. Must be provided.        |
|               |            | This field is required.                            |
|               |            |                                                    |
+---------------+------------+----------------------------------------------------+
| allow_partial | ``BOOL``   | Whether to allow partial traces. Default to False. |
+---------------+------------+----------------------------------------------------+

.. _mlflowGetTraceInfo:

GetTraceInfo
------------






+------------+------------+---------------------------------------------+
| Field Name |    Type    |                 Description                 |
+============+============+=============================================+
| request_id | ``STRING`` | ID of the trace to fetch. Must be provided. |
+------------+------------+---------------------------------------------+

.. _mlflowGetTraceInfoV3:

GetTraceInfoV3
--------------






+------------+------------+---------------------------------------------+
| Field Name |    Type    |                 Description                 |
+============+============+=============================================+
| trace_id   | ``STRING`` | ID of the trace to fetch. Must be provided. |
+------------+------------+---------------------------------------------+

.. _mlflowartifactsMultipartUploadCredentialHeadersEntry:

HeadersEntry
------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowTraceLocationInferenceTableLocation:

InferenceTableLocation
----------------------






+-----------------+------------+--------------------------------------------------------------------+
|   Field Name    |    Type    |                            Description                             |
+=================+============+====================================================================+
| full_table_name | ``STRING`` | Full inference table name in the form of catalog.schema.table_name |
+-----------------+------------+--------------------------------------------------------------------+

.. _mlflowPromptOptimizationJobInitialEvalScoresEntry:

InitialEvalScoresEntry
----------------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``DOUBLE`` |             |
+------------+------------+-------------+

.. _mlflowInputTag:

InputTag
--------



Tag for an input.


+------------+------------+-------------------------+
| Field Name |    Type    |       Description       |
+============+============+=========================+
| key        | ``STRING`` | The tag key.            |
|            |            | This field is required. |
|            |            |                         |
+------------+------------+-------------------------+
| value      | ``STRING`` | The tag value.          |
|            |            | This field is required. |
|            |            |                         |
+------------+------------+-------------------------+

.. _mlflowJobState:

JobState
--------



Generic job state message combining status with metadata.
Provides a unified way to represent job state across different job types.


+---------------+------------------------------------------------+------------------------------------------------------+
|  Field Name   |                      Type                      |                     Description                      |
+===============+================================================+======================================================+
| status        | :ref:`mlflowjobstatus`                         | Current status of the job.                           |
+---------------+------------------------------------------------+------------------------------------------------------+
| error_message | ``STRING``                                     | Error message if the job failed.                     |
|               |                                                | Only set when status is JOB_STATUS_FAILED.           |
+---------------+------------------------------------------------+------------------------------------------------------+
| metadata      | An array of :ref:`mlflowjobstatemetadataentry` | Additional metadata as key-value pairs.              |
|               |                                                | Can be used to store job-specific state information. |
+---------------+------------------------------------------------+------------------------------------------------------+

.. _mlflowLinkPromptsToTrace:

LinkPromptsToTrace
------------------






+-----------------+-------------------------------------------------------------+---------------------------------------------+
|   Field Name    |                            Type                             |                 Description                 |
+=================+=============================================================+=============================================+
| trace_id        | ``STRING``                                                  | ID of the trace to link prompt versions to. |
|                 |                                                             | This field is required.                     |
|                 |                                                             |                                             |
+-----------------+-------------------------------------------------------------+---------------------------------------------+
| prompt_versions | An array of :ref:`mlflowlinkpromptstotracepromptversionref` |                                             |
+-----------------+-------------------------------------------------------------+---------------------------------------------+

.. _mlflowLinkTracesToRun:

LinkTracesToRun
---------------






+------------+------------------------+--------------------------------------------------------------------------------+
| Field Name |          Type          |                                  Description                                   |
+============+========================+================================================================================+
| trace_ids  | An array of ``STRING`` | IDs of the traces to link to the run.                                          |
|            |                        | The maximum number of trace IDs that can be linked in a single request is 100. |
+------------+------------------------+--------------------------------------------------------------------------------+
| run_id     | ``STRING``             | ID of the run to link the traces to.                                           |
|            |                        | This field is required.                                                        |
|            |                        |                                                                                |
+------------+------------------------+--------------------------------------------------------------------------------+

.. _mlflowListLoggedModelArtifacts:

ListLoggedModelArtifacts
------------------------






+-------------------------+------------+-----------------------------------------------------------------------------------------+
|       Field Name        |    Type    |                                       Description                                       |
+=========================+============+=========================================================================================+
| model_id                | ``STRING`` | The ID of the LoggedModel for which to list the artifacts                               |
|                         |            | This field is required.                                                                 |
|                         |            |                                                                                         |
+-------------------------+------------+-----------------------------------------------------------------------------------------+
| artifact_directory_path | ``STRING`` | Filter artifacts matching this path (a relative path from the root artifact directory). |
+-------------------------+------------+-----------------------------------------------------------------------------------------+
| page_token              | ``STRING`` | Token indicating the page of artifact results to fetch                                  |
+-------------------------+------------+-----------------------------------------------------------------------------------------+

.. _mlflowLogLoggedModelParamsRequest:

LogLoggedModelParamsRequest
---------------------------






+------------+-----------------------------------------------+-----------------------------------------------+
| Field Name |                     Type                      |                  Description                  |
+============+===============================================+===============================================+
| model_id   | ``STRING``                                    | The ID of the logged model to log params for. |
|            |                                               | This field is required.                       |
|            |                                               |                                               |
+------------+-----------------------------------------------+-----------------------------------------------+
| params     | An array of :ref:`mlflowloggedmodelparameter` | Parameters attached to the model.             |
+------------+-----------------------------------------------+-----------------------------------------------+

.. _mlflowLogOutputs:

LogOutputs
----------






+------------+--------------------------------------+------------------------------------------+
| Field Name |                 Type                 |               Description                |
+============+======================================+==========================================+
| run_id     | ``STRING``                           | ID of the Run from which to log outputs. |
|            |                                      | This field is required.                  |
|            |                                      |                                          |
+------------+--------------------------------------+------------------------------------------+
| models     | An array of :ref:`mlflowmodeloutput` | Model outputs from the Run.              |
+------------+--------------------------------------+------------------------------------------+

.. _mlflowLoggedModel:

LoggedModel
-----------



A LoggedModel message includes logged model attributes,
tags, registration info, params, and linked run metrics.


+------------+------------------------------+-------------------------------------------------------------+
| Field Name |             Type             |                         Description                         |
+============+==============================+=============================================================+
| info       | :ref:`mlflowloggedmodelinfo` | LoggedModel attributes such as model ID, status, tags, etc. |
+------------+------------------------------+-------------------------------------------------------------+
| data       | :ref:`mlflowloggedmodeldata` | LoggedModel params and metrics.                             |
+------------+------------------------------+-------------------------------------------------------------+

.. _mlflowLoggedModelData:

LoggedModelData
---------------



A LoggedModelData message includes logged model params and linked metrics.


+------------+-----------------------------------------------+------------------------------------------------+
| Field Name |                     Type                      |                  Description                   |
+============+===============================================+================================================+
| params     | An array of :ref:`mlflowloggedmodelparameter` | Immutable String key-value pairs of the model. |
+------------+-----------------------------------------------+------------------------------------------------+
| metrics    | An array of :ref:`mlflowmetric`               | Performance metrics linked to the model.       |
+------------+-----------------------------------------------+------------------------------------------------+

.. _mlflowLoggedModelInfo:

LoggedModelInfo
---------------



A LoggedModelInfo includes logged model attributes,
tags, and registration info.


+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
|        Field Name         |                         Type                         |                                   Description                                   |
+===========================+======================================================+=================================================================================+
| model_id                  | ``STRING``                                           | A unique identifier for the model.                                              |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| experiment_id             | ``STRING``                                           | The ID of the experiment that owns the model.                                   |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| name                      | ``STRING``                                           | Name of the model.                                                              |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| creation_timestamp_ms     | ``INT64``                                            | Timestamp when the model was created, in milliseconds since the UNIX epoch.     |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| last_updated_timestamp_ms | ``INT64``                                            | Timestamp when the model was last updated, in milliseconds since the UNIX epoch |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| artifact_uri              | ``STRING``                                           | URI of the directory where model artifacts are stored.                          |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| status                    | :ref:`mlflowloggedmodelstatus`                       | Whether or not the model is ready for use.                                      |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| creator_id                | ``INT64``                                            | The ID of the user or principal that created the model.                         |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| model_type                | ``STRING``                                           | The type of model, such as "Agent", "Classifier", "LLM".                        |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| source_run_id             | ``STRING``                                           | Run ID of the run that created the model.                                       |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| status_message            | ``STRING``                                           | Details on the current status.                                                  |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| tags                      | An array of :ref:`mlflowloggedmodeltag`              | Mutable String key-value pairs set on the model.                                |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+
| registrations             | An array of :ref:`mlflowloggedmodelregistrationinfo` | If the model has been promoted to the Model Registry, this field includes       |
|                           |                                                      | information like the Registered Model name, Model Version number, etc.          |
+---------------------------+------------------------------------------------------+---------------------------------------------------------------------------------+

.. _mlflowLoggedModelParameter:

LoggedModelParameter
--------------------



Parameter associated with a LoggedModel.


+------------+------------+-----------------------------------+
| Field Name |    Type    |            Description            |
+============+============+===================================+
| key        | ``STRING`` | Key identifying this param.       |
+------------+------------+-----------------------------------+
| value      | ``STRING`` | Value associated with this param. |
+------------+------------+-----------------------------------+

.. _mlflowLoggedModelRegistrationInfo:

LoggedModelRegistrationInfo
---------------------------



RegistrationInfo for a LoggedModel.


+------------+------------+------------------------------------------------------------------------+
| Field Name |    Type    |                              Description                               |
+============+============+========================================================================+
| name       | ``STRING`` | The name of the Registered Model to which the model has been promoted. |
+------------+------------+------------------------------------------------------------------------+
| version    | ``STRING`` | The version number of the promoted model.                              |
+------------+------------+------------------------------------------------------------------------+

.. _mlflowLoggedModelTag:

LoggedModelTag
--------------



Tag for a LoggedModel.


+------------+------------+----------------+
| Field Name |    Type    |  Description   |
+============+============+================+
| key        | ``STRING`` | The tag key.   |
+------------+------------+----------------+
| value      | ``STRING`` | The tag value. |
+------------+------------+----------------+

.. _mlflowGatewaySecretInfoMaskedValuesEntry:

MaskedValuesEntry
-----------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowassessmentsAssessmentMetadataEntry:

MetadataEntry
-------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowJobStateMetadataEntry:

MetadataEntry
-------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowMetric:

Metric
------



Metric associated with a run, represented as a key-value pair.


+----------------+------------+-----------------------------------------------------------------------+
|   Field Name   |    Type    |                              Description                              |
+================+============+=======================================================================+
| key            | ``STRING`` | Key identifying this metric.                                          |
+----------------+------------+-----------------------------------------------------------------------+
| value          | ``DOUBLE`` | Value associated with this metric.                                    |
+----------------+------------+-----------------------------------------------------------------------+
| timestamp      | ``INT64``  | The timestamp at which this metric was recorded.                      |
+----------------+------------+-----------------------------------------------------------------------+
| step           | ``INT64``  | Step at which to log the metric.                                      |
+----------------+------------+-----------------------------------------------------------------------+
| dataset_name   | ``STRING`` | The name of the dataset associated with the metric.                   |
|                |            | E.g. "my.uc.table@2" "nyc-taxi-dataset", "fantastic-elk-3"            |
+----------------+------------+-----------------------------------------------------------------------+
| dataset_digest | ``STRING`` | Dataset digest of the dataset associated with the metric,             |
|                |            | e.g. an md5 hash of the dataset that uniquely identifies it           |
|                |            | within datasets of the same name.                                     |
+----------------+------------+-----------------------------------------------------------------------+
| model_id       | ``STRING`` | The ID of the LoggedModel or Registered Model Version associated with |
|                |            | the metric, if applicable.                                            |
+----------------+------------+-----------------------------------------------------------------------+
| run_id         | ``STRING`` | The ID of the run containing the metric.                              |
+----------------+------------+-----------------------------------------------------------------------+

.. _mlflowMetricAggregation:

MetricAggregation
-----------------






+------------------+------------------------------+----------------------------------------------------------------------------------------+
|    Field Name    |             Type             |                                      Description                                       |
+==================+==============================+========================================================================================+
| aggregation_type | :ref:`mlflowaggregationtype` | The type of aggregation to perform.                                                    |
+------------------+------------------------------+----------------------------------------------------------------------------------------+
| percentile_value | ``DOUBLE``                   | The percentile value to compute (0-100), required when aggregation_type is PERCENTILE. |
|                  |                              | Examples: 50 (median), 75, 90, 95, 99.                                                 |
|                  |                              | This field is ignored for other aggregation types.                                     |
+------------------+------------------------------+----------------------------------------------------------------------------------------+

.. _mlflowMetricDataPoint:

MetricDataPoint
---------------



A single data point with dimension values and metric values.


+-------------+---------------------------------------------------------+--------------------------------------+
| Field Name  |                          Type                           |             Description              |
+=============+=========================================================+======================================+
| metric_name | ``STRING``                                              | Metric name, e.g. "latency"          |
+-------------+---------------------------------------------------------+--------------------------------------+
| dimensions  | An array of :ref:`mlflowmetricdatapointdimensionsentry` | Dimension values for this data point |
|             |                                                         | Keys correspond to dimensions        |
|             |                                                         | e.g., {"status": "OK"}               |
+-------------+---------------------------------------------------------+--------------------------------------+
| values      | An array of :ref:`mlflowmetricdatapointvaluesentry`     | Metric values for this data point    |
|             |                                                         | Keys are aggregation types           |
|             |                                                         | e.g., {"AVG": 150, "P99": 234.5}     |
+-------------+---------------------------------------------------------+--------------------------------------+

.. _mlflowMetricWithRunId:

MetricWithRunId
---------------






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
| run_id     | ``STRING`` | The ID of the run containing the metric          |
+------------+------------+--------------------------------------------------+

.. _mlflowTraceLocationMlflowExperimentLocation:

MlflowExperimentLocation
------------------------






+---------------+------------+--------------------------------------------------------------------+
|  Field Name   |    Type    |                            Description                             |
+===============+============+====================================================================+
| experiment_id | ``STRING`` | MLflow experiment ID which is the ACL container holding the trace. |
+---------------+------------+--------------------------------------------------------------------+

.. _mlflowModelInput:

ModelInput
----------



Represents a LoggedModel or Registered Model Version input to a Run.


+------------+------------+-------------------------------------+
| Field Name |    Type    |             Description             |
+============+============+=====================================+
| model_id   | ``STRING`` | The unique identifier of the model. |
|            |            | This field is required.             |
|            |            |                                     |
+------------+------------+-------------------------------------+

.. _mlflowModelMetric:

ModelMetric
-----------



Metric associated with a model, represented as a key-value pair.
Copied from MLflow metric


+----------------+------------+-----------------------------------------------------------------------+
|   Field Name   |    Type    |                              Description                              |
+================+============+=======================================================================+
| key            | ``STRING`` | Key identifying this metric.                                          |
+----------------+------------+-----------------------------------------------------------------------+
| value          | ``DOUBLE`` | Value associated with this metric.                                    |
+----------------+------------+-----------------------------------------------------------------------+
| timestamp      | ``INT64``  | The timestamp at which this metric was recorded.                      |
+----------------+------------+-----------------------------------------------------------------------+
| step           | ``INT64``  | Step at which to log the metric.                                      |
+----------------+------------+-----------------------------------------------------------------------+
| dataset_name   | ``STRING`` | The name of the dataset associated with the metric.                   |
|                |            | E.g. "my.uc.table@2" "nyc-taxi-dataset", "fantastic-elk-3"            |
+----------------+------------+-----------------------------------------------------------------------+
| dataset_digest | ``STRING`` | Dataset digest of the dataset associated with the metric,             |
|                |            | e.g. an md5 hash of the dataset that uniquely identifies it           |
|                |            | within datasets of the same name.                                     |
+----------------+------------+-----------------------------------------------------------------------+
| model_id       | ``STRING`` | The ID of the LoggedModel or Registered Model Version associated with |
|                |            | the metric                                                            |
+----------------+------------+-----------------------------------------------------------------------+
| run_id         | ``STRING`` | The ID of the run containing the metric.                              |
+----------------+------------+-----------------------------------------------------------------------+

.. _mlflowModelOutput:

ModelOutput
-----------



Represents a LoggedModel output of a Run.


+------------+------------+---------------------------------------+
| Field Name |    Type    |              Description              |
+============+============+=======================================+
| model_id   | ``STRING`` | The unique identifier of the model.   |
|            |            | This field is required.               |
|            |            |                                       |
+------------+------------+---------------------------------------+
| step       | ``INT64``  | Step at which the model was produced. |
|            |            | This field is required.               |
|            |            |                                       |
+------------+------------+---------------------------------------+

.. _mlflowModelParam:

ModelParam
----------



Param for a model version.


+------------+------------+-------------------------------------------------------------+
| Field Name |    Type    |                         Description                         |
+============+============+=============================================================+
| name       | ``STRING`` | Name of the param.                                          |
+------------+------------+-------------------------------------------------------------+
| value      | ``STRING`` | Value of the param associated with the name, could be empty |
+------------+------------+-------------------------------------------------------------+

.. _mlflowModelVersion:

ModelVersion
------------






+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
|       Field Name       |                    Type                     |                                                  Description                                                   |
+========================+=============================================+================================================================================================================+
| name                   | ``STRING``                                  | Unique name of the model                                                                                       |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| version                | ``STRING``                                  | Model's version number.                                                                                        |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| creation_timestamp     | ``INT64``                                   | Timestamp recorded when this ``model_version`` was created.                                                    |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| last_updated_timestamp | ``INT64``                                   | Timestamp recorded when metadata for this ``model_version`` was last updated.                                  |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| user_id                | ``STRING``                                  | User that created this ``model_version``.                                                                      |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| current_stage          | ``STRING``                                  | Current stage for this ``model_version``.                                                                      |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| description            | ``STRING``                                  | Description of this ``model_version``.                                                                         |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| source                 | ``STRING``                                  | URI indicating the location of the source model artifacts, used when creating ``model_version``                |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| run_id                 | ``STRING``                                  | MLflow run ID used when creating ``model_version``, if ``source`` was generated by an                          |
|                        |                                             | experiment run stored in MLflow tracking server.                                                               |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| status                 | :ref:`mlflowmodelversionstatus`             | Current status of ``model_version``                                                                            |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| status_message         | ``STRING``                                  | Details on current ``status``, if it is pending or failed.                                                     |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| tags                   | An array of :ref:`mlflowmodelversiontag`    | Tags: Additional metadata key-value pairs for this ``model_version``.                                          |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| run_link               | ``STRING``                                  | Run Link: Direct link to the run that generated this version. This field is set at model version creation time |
|                        |                                             | only for model versions whose source run is from a tracking server that is different from the registry server. |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| aliases                | An array of ``STRING``                      | Aliases pointing to this ``model_version``.                                                                    |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| model_id               | ``STRING``                                  | Optional `model_id` for model version that is used to link the registered model to the source logged model     |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| model_params           | An array of :ref:`mlflowmodelparam`         | Optional parameters for the model.                                                                             |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| model_metrics          | An array of :ref:`mlflowmodelmetric`        | Optional metrics for the model.                                                                                |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| deployment_job_state   | :ref:`mlflowmodelversiondeploymentjobstate` | Deployment job state for this model version.                                                                   |
+------------------------+---------------------------------------------+----------------------------------------------------------------------------------------------------------------+

.. _mlflowModelVersionDeploymentJobState:

ModelVersionDeploymentJobState
------------------------------






+-------------------+------------------------------------------------------------------+-------------+
|    Field Name     |                               Type                               | Description |
+===================+==================================================================+=============+
| job_id            | ``STRING``                                                       |             |
+-------------------+------------------------------------------------------------------+-------------+
| run_id            | ``STRING``                                                       |             |
+-------------------+------------------------------------------------------------------+-------------+
| job_state         | :ref:`mlflowdeploymentjobconnectionstate`                        |             |
+-------------------+------------------------------------------------------------------+-------------+
| run_state         | :ref:`mlflowmodelversiondeploymentjobstatedeploymentjobrunstate` |             |
+-------------------+------------------------------------------------------------------+-------------+
| current_task_name | ``STRING``                                                       |             |
+-------------------+------------------------------------------------------------------+-------------+

.. _mlflowModelVersionTag:

ModelVersionTag
---------------



Tag for a model version.


+------------+------------+----------------+
| Field Name |    Type    |  Description   |
+============+============+================+
| key        | ``STRING`` | The tag key.   |
+------------+------------+----------------+
| value      | ``STRING`` | The tag value. |
+------------+------------+----------------+

.. _mlflowartifactsMultipartUploadCredential:

MultipartUploadCredential
-------------------------






+-------------+-------------------------------------------------------------------------+-------------+
| Field Name  |                                  Type                                   | Description |
+=============+=========================================================================+=============+
| url         | ``STRING``                                                              |             |
+-------------+-------------------------------------------------------------------------+-------------+
| part_number | ``INT64``                                                               |             |
+-------------+-------------------------------------------------------------------------+-------------+
| headers     | An array of :ref:`mlflowartifactsmultipartuploadcredentialheadersentry` |             |
+-------------+-------------------------------------------------------------------------+-------------+

.. _mlflowartifactsMultipartUploadPart:

MultipartUploadPart
-------------------






+-------------+------------+-------------+
| Field Name  |    Type    | Description |
+=============+============+=============+
| part_number | ``INT64``  |             |
+-------------+------------+-------------+
| etag        | ``STRING`` |             |
+-------------+------------+-------------+
| url         | ``STRING`` |             |
+-------------+------------+-------------+

.. _mlflowSearchLoggedModelsOrderBy:

OrderBy
-------






+----------------+------------+---------------------------------------------------------------------------------------------+
|   Field Name   |    Type    |                                         Description                                         |
+================+============+=============================================================================================+
| field_name     | ``STRING`` | Name of the field to order by, e.g. "metrics.accuracy".                                     |
|                |            | This field is required.                                                                     |
|                |            |                                                                                             |
+----------------+------------+---------------------------------------------------------------------------------------------+
| ascending      | ``BOOL``   | Whether the order is ascending or not.                                                      |
+----------------+------------+---------------------------------------------------------------------------------------------+
| dataset_name   | ``STRING`` | If ``field_name`` refers to a metric, this field specifies the name of the dataset          |
|                |            | associated with the metric. Only metrics associated with the specified dataset name will be |
|                |            | considered for ordering. This field may only be set if ``field_name`` refers to a metric.   |
+----------------+------------+---------------------------------------------------------------------------------------------+
| dataset_digest | ``STRING`` | If ``field_name`` refers to a metric, this field specifies the digest of the dataset        |
|                |            | associated with the metric. Only metrics associated with the specified dataset name         |
|                |            | and digest will be considered for ordering. This field may only be set if ``dataset_name``  |
|                |            | is also set.                                                                                |
+----------------+------------+---------------------------------------------------------------------------------------------+

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

.. _mlflowPromptOptimizationJob:

PromptOptimizationJob
---------------------



Represents a prompt optimization job entity.


+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|       Field Name        |                                 Type                                 |                                       Description                                       |
+=========================+======================================================================+=========================================================================================+
| job_id                  | ``STRING``                                                           | Unique identifier for the optimization job.                                             |
|                         |                                                                      | Used to poll job execution status (pending/running/completed/failed).                   |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| run_id                  | ``STRING``                                                           | MLflow run ID where optimization metrics and results are stored.                        |
|                         |                                                                      | Use this to view results in MLflow UI. Only available after job starts running.         |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| state                   | :ref:`mlflowjobstate`                                                | Current state of the job (status + error message + metadata).                           |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| experiment_id           | ``STRING``                                                           | ID of the MLflow experiment where this optimization job is tracked.                     |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| source_prompt_uri       | ``STRING``                                                           | URI of the source prompt that optimization started from (e.g., "prompts:/my-prompt/1"). |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| optimized_prompt_uri    | ``STRING``                                                           | URI of the optimized prompt (e.g., "prompts:/my-prompt/2").                             |
|                         |                                                                      | Only set if optimization completed successfully.                                        |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| config                  | :ref:`mlflowpromptoptimizationjobconfig`                             | Configuration for the optimization job.                                                 |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| creation_timestamp_ms   | ``INT64``                                                            | Timestamp when the job was created (milliseconds since epoch).                          |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| completion_timestamp_ms | ``INT64``                                                            | Timestamp when the job completed (milliseconds since epoch).                            |
|                         |                                                                      | Only set if status is COMPLETED, FAILED, or CANCELED.                                   |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| tags                    | An array of :ref:`mlflowpromptoptimizationjobtag`                    | Tags associated with this job.                                                          |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| initial_eval_scores     | An array of :ref:`mlflowpromptoptimizationjobinitialevalscoresentry` | Initial evaluation scores before optimization, keyed by scorer name.                    |
|                         |                                                                      | Example: {"Correctness": 0.65, "Safety": 0.80}                                          |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| final_eval_scores       | An array of :ref:`mlflowpromptoptimizationjobfinalevalscoresentry`   | Final evaluation scores after optimization, keyed by scorer name.                       |
|                         |                                                                      | Example: {"Correctness": 0.89, "Safety": 0.95}                                          |
+-------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------------+

.. _mlflowPromptOptimizationJobConfig:

PromptOptimizationJobConfig
---------------------------



Configuration for a prompt optimization job.
Stored as run parameters in the underlying MLflow run.


+-----------------------+----------------------------+----------------------------------------------------------------------------------------------+
|      Field Name       |            Type            |                                         Description                                          |
+=======================+============================+==============================================================================================+
| optimizer_type        | :ref:`mlflowoptimizertype` | The optimizer type to use.                                                                   |
+-----------------------+----------------------------+----------------------------------------------------------------------------------------------+
| dataset_id            | ``STRING``                 | ID of the EvaluationDataset containing training data.                                        |
+-----------------------+----------------------------+----------------------------------------------------------------------------------------------+
| scorers               | An array of ``STRING``     | List of scorer names. Can be built-in scorer class names                                     |
|                       |                            | (e.g., "Correctness", "Safety") or registered scorer names.                                  |
+-----------------------+----------------------------+----------------------------------------------------------------------------------------------+
| optimizer_config_json | ``STRING``                 | JSON-serialized optimizer-specific configuration.                                            |
|                       |                            | Different optimizers accept different parameters:                                            |
|                       |                            | - GEPA: {"reflection_model": "openai:/gpt-5", "max_metric_calls": 300}                       |
|                       |                            | - MetaPrompt: {"reflection_model": "openai:/gpt-5", "guidelines": "...", "lm_kwargs": {...}} |
+-----------------------+----------------------------+----------------------------------------------------------------------------------------------+

.. _mlflowPromptOptimizationJobTag:

PromptOptimizationJobTag
------------------------



Tag for a prompt optimization job.


+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowLinkPromptsToTracePromptVersionRef:

PromptVersionRef
----------------



Prompt version references to link to the trace.
Each reference contains the prompt name and version.


+------------+------------+-------------------------+
| Field Name |    Type    |       Description       |
+============+============+=========================+
| name       | ``STRING`` |                         |
|            |            | This field is required. |
|            |            |                         |
+------------+------------+-------------------------+
| version    | ``STRING`` |                         |
|            |            | This field is required. |
|            |            |                         |
+------------+------------+-------------------------+

.. _mlflowQueryTraceMetrics:

QueryTraceMetrics
-----------------



Query aggregated metrics for traces, spans, or assessments.


+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+
|      Field Name       |                    Type                    |                                   Description                                    |
+=======================+============================================+==================================================================================+
| experiment_ids        | An array of ``STRING``                     | Required: The experiment IDs to search traces.                                   |
+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| view_type             | :ref:`mlflowmetricviewtype`                | Required: The level at which to aggregate metrics.                               |
+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| metric_name           | ``STRING``                                 | Required: The name of the metric to query (e.g. "latency").                      |
+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| aggregations          | An array of :ref:`mlflowmetricaggregation` | Required: The aggregations to apply.                                             |
+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| dimensions            | An array of ``STRING``                     | Optional: Dimensions to group metrics by. (e.g. "name", "status")                |
+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| filters               | An array of ``STRING``                     | Optional: Filter expressions to apply. (e.g. `trace.status="OK"`)                |
+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| time_interval_seconds | ``INT64``                                  | Optional: Time interval for grouping in seconds.                                 |
|                       |                                            | When set, results automatically include a time dimension grouped by              |
|                       |                                            | the specified interval.                                                          |
|                       |                                            | Examples: 60 (minute), 3600 (hour), 86400 (day), 604800 (week), 2592000 (month). |
+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| start_time_ms         | ``INT64``                                  | Optional: Start of time range in milliseconds since epoch.                       |
|                       |                                            | Required if time_interval_seconds is set.                                        |
+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| end_time_ms           | ``INT64``                                  | Optional: End of time range in milliseconds since epoch.                         |
|                       |                                            | Required if time_interval_seconds is set.                                        |
+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| max_results           | ``INT32``                                  | Optional: Maximum number of data points to return.                               |
|                       |                                            | Default: 1000                                                                    |
+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+
| page_token            | ``STRING``                                 | Optional: Pagination token for fetching the next page of results.                |
+-----------------------+--------------------------------------------+----------------------------------------------------------------------------------+

.. _mlflowRegisteredModel:

RegisteredModel
---------------






+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
|       Field Name       |                     Type                      |                                   Description                                    |
+========================+===============================================+==================================================================================+
| name                   | ``STRING``                                    | Unique name for the model.                                                       |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| creation_timestamp     | ``INT64``                                     | Timestamp recorded when this ``registered_model`` was created.                   |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| last_updated_timestamp | ``INT64``                                     | Timestamp recorded when metadata for this ``registered_model`` was last updated. |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| user_id                | ``STRING``                                    | User that created this ``registered_model``                                      |
|                        |                                               | NOTE: this field is not currently returned.                                      |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| description            | ``STRING``                                    | Description of this ``registered_model``.                                        |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| latest_versions        | An array of :ref:`mlflowmodelversion`         | Collection of latest model versions for each stage.                              |
|                        |                                               | Only contains models with current ``READY`` status.                              |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| tags                   | An array of :ref:`mlflowregisteredmodeltag`   | Tags: Additional metadata key-value pairs for this ``registered_model``.         |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| aliases                | An array of :ref:`mlflowregisteredmodelalias` | Aliases pointing to model versions associated with this ``registered_model``.    |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| deployment_job_id      | ``STRING``                                    | Deployment job id for this model.                                                |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+
| deployment_job_state   | :ref:`mlflowdeploymentjobconnectionstate`     | Deployment job state for this model.                                             |
+------------------------+-----------------------------------------------+----------------------------------------------------------------------------------+

.. _mlflowRegisteredModelAlias:

RegisteredModelAlias
--------------------



Alias for a registered model


+------------+------------+----------------------------------------------------+
| Field Name |    Type    |                    Description                     |
+============+============+====================================================+
| alias      | ``STRING`` | The name of the alias.                             |
+------------+------------+----------------------------------------------------+
| version    | ``STRING`` | The model version number that the alias points to. |
+------------+------------+----------------------------------------------------+

.. _mlflowRegisteredModelTag:

RegisteredModelTag
------------------



Tag for a registered model


+------------+------------+----------------+
| Field Name |    Type    |  Description   |
+============+============+================+
| key        | ``STRING`` | The tag key.   |
+------------+------------+----------------+
| value      | ``STRING`` | The tag value. |
+------------+------------+----------------+

.. _mlflowRemoveDatasetFromExperiments:

RemoveDatasetFromExperiments
----------------------------






+----------------+------------------------+-------------------------------------------------+
|   Field Name   |          Type          |                   Description                   |
+================+========================+=================================================+
| dataset_id     | ``STRING``             | Dataset ID to remove from experiments           |
|                |                        | This field is required.                         |
|                |                        |                                                 |
+----------------+------------------------+-------------------------------------------------+
| experiment_ids | An array of ``STRING`` | Experiment IDs to disassociate from the dataset |
+----------------+------------------------+-------------------------------------------------+

.. _mlflowGetMetricHistoryBulkIntervalResponse:

Response
--------






+------------+------------------------------------------+--------------------------------------------------------------+
| Field Name |                   Type                   |                         Description                          |
+============+==========================================+==============================================================+
| metrics    | An array of :ref:`mlflowmetricwithrunid` | List of metrics representing history of values and metadata. |
+------------+------------------------------------------+--------------------------------------------------------------+

.. _mlflowCreateAssessmentResponse:

Response
--------






+------------+------------------------------------+-------------------------+
| Field Name |                Type                |       Description       |
+============+====================================+=========================+
| assessment | :ref:`mlflowassessmentsassessment` | The created assessment. |
+------------+------------------------------------+-------------------------+

.. _mlflowUpdateAssessmentResponse:

Response
--------






+------------+------------------------------------+----------------------------------+
| Field Name |                Type                |           Description            |
+============+====================================+==================================+
| assessment | :ref:`mlflowassessmentsassessment` | The Assessment after the update. |
+------------+------------------------------------+----------------------------------+

.. _mlflowGetAssessmentRequestResponse:

Response
--------






+------------+------------------------------------+---------------------------+
| Field Name |                Type                |        Description        |
+============+====================================+===========================+
| assessment | :ref:`mlflowassessmentsassessment` | The requested assessment. |
+------------+------------------------------------+---------------------------+

.. _mlflowStartTraceResponse:

Response
--------






+------------+------------------------+--------------------------+
| Field Name |          Type          |       Description        |
+============+========================+==========================+
| trace_info | :ref:`mlflowtraceinfo` | The newly created trace. |
+------------+------------------------+--------------------------+

.. _mlflowEndTraceResponse:

Response
--------






+------------+------------------------+--------------------+
| Field Name |          Type          |    Description     |
+============+========================+====================+
| trace_info | :ref:`mlflowtraceinfo` | The updated trace. |
+------------+------------------------+--------------------+

.. _mlflowGetTraceInfoResponse:

Response
--------






+------------+------------------------+----------------------------------+
| Field Name |          Type          |           Description            |
+============+========================+==================================+
| trace_info | :ref:`mlflowtraceinfo` | Metadata of the requested trace. |
+------------+------------------------+----------------------------------+

.. _mlflowGetTraceInfoV3Response:

Response
--------






+------------+--------------------+-------------+
| Field Name |        Type        | Description |
+============+====================+=============+
| trace      | :ref:`mlflowtrace` |             |
+------------+--------------------+-------------+

.. _mlflowBatchGetTracesResponse:

Response
--------






+------------+--------------------------------+--------------------------------+
| Field Name |              Type              |          Description           |
+============+================================+================================+
| traces     | An array of :ref:`mlflowtrace` | The fetched trace information. |
+------------+--------------------------------+--------------------------------+

.. _mlflowGetTraceResponse:

Response
--------






+------------+--------------------+------------------------------------+
| Field Name |        Type        |            Description             |
+============+====================+====================================+
| trace      | :ref:`mlflowtrace` | The fetched trace including spans. |
+------------+--------------------+------------------------------------+

.. _mlflowSearchTracesResponse:

Response
--------






+-----------------+------------------------------------+----------------------------------------------------------+
|   Field Name    |                Type                |                       Description                        |
+=================+====================================+==========================================================+
| traces          | An array of :ref:`mlflowtraceinfo` | Information about traces that match the search criteria. |
+-----------------+------------------------------------+----------------------------------------------------------+
| next_page_token | ``STRING``                         |                                                          |
+-----------------+------------------------------------+----------------------------------------------------------+

.. _mlflowSearchUnifiedTracesResponse:

Response
--------






+-----------------+------------------------------------+----------------------------------------------------------+
|   Field Name    |                Type                |                       Description                        |
+=================+====================================+==========================================================+
| traces          | An array of :ref:`mlflowtraceinfo` | Information about traces that match the search criteria. |
+-----------------+------------------------------------+----------------------------------------------------------+
| next_page_token | ``STRING``                         |                                                          |
+-----------------+------------------------------------+----------------------------------------------------------+

.. _mlflowGetOnlineTraceDetailsResponse:

Response
--------






+------------+------------+--------------------------------------------------------------+
| Field Name |    Type    |                         Description                          |
+============+============+==============================================================+
| trace_data | ``STRING`` | Return trace JSON in string form                             |
|            |            | Note: we may change this to a TraceData object in the future |
+------------+------------+--------------------------------------------------------------+

.. _mlflowDeleteTracesResponse:

Response
--------






+----------------+-----------+-------------+
|   Field Name   |   Type    | Description |
+================+===========+=============+
| traces_deleted | ``INT32`` |             |
+----------------+-----------+-------------+

.. _mlflowDeleteTracesV3Response:

Response
--------






+----------------+-----------+-------------+
|   Field Name   |   Type    | Description |
+================+===========+=============+
| traces_deleted | ``INT32`` |             |
+----------------+-----------+-------------+

.. _mlflowCalculateTraceFilterCorrelationResponse:

Response
--------






+---------------+------------+----------------------------------------------------------+
|  Field Name   |    Type    |                       Description                        |
+===============+============+==========================================================+
| npmi          | ``DOUBLE`` | Normalized Pointwise Mutual Information score (-1 to 1). |
+---------------+------------+----------------------------------------------------------+
| npmi_smoothed | ``DOUBLE`` | Smoothed NPMI value with Jeffreys prior for robustness.  |
+---------------+------------+----------------------------------------------------------+
| filter1_count | ``INT32``  | Number of traces matching the first filter.              |
+---------------+------------+----------------------------------------------------------+
| filter2_count | ``INT32``  | Number of traces matching the second filter.             |
+---------------+------------+----------------------------------------------------------+
| joint_count   | ``INT32``  | Number of traces matching both filters.                  |
+---------------+------------+----------------------------------------------------------+
| total_count   | ``INT32``  | Total number of traces in the experiments.               |
+---------------+------------+----------------------------------------------------------+

.. _mlflowQueryTraceMetricsResponse:

Response
--------






+-----------------+------------------------------------------+----------------------------------------------+
|   Field Name    |                   Type                   |                 Description                  |
+=================+==========================================+==============================================+
| data_points     | An array of :ref:`mlflowmetricdatapoint` | Data points grouped by dimensions.           |
+-----------------+------------------------------------------+----------------------------------------------+
| next_page_token | ``STRING``                               | Pagination token for fetching the next page. |
|                 |                                          | Empty if no more results are available.      |
+-----------------+------------------------------------------+----------------------------------------------+

.. _mlflowStartTraceV3Response:

Response
--------






+------------+--------------------+--------------------------------+
| Field Name |        Type        |          Description           |
+============+====================+================================+
| trace      | :ref:`mlflowtrace` | The created trace information. |
+------------+--------------------+--------------------------------+

.. _mlflowSearchDatasetsResponse:

Response
--------






+-------------------+-----------------------------------------+-----------------------------------------------------------------------------------+
|    Field Name     |                  Type                   |                                    Description                                    |
+===================+=========================================+===================================================================================+
| dataset_summaries | An array of :ref:`mlflowdatasetsummary` | Return the summary for most recently created N datasets, as configured in backend |
+-------------------+-----------------------------------------+-----------------------------------------------------------------------------------+

.. _mlflowCreateLoggedModelResponse:

Response
--------






+------------+--------------------------+--------------------------------+
| Field Name |           Type           |          Description           |
+============+==========================+================================+
| model      | :ref:`mlflowloggedmodel` | The newly created LoggedModel. |
+------------+--------------------------+--------------------------------+

.. _mlflowFinalizeLoggedModelResponse:

Response
--------






+------------+--------------------------+--------------------------+
| Field Name |           Type           |       Description        |
+============+==========================+==========================+
| model      | :ref:`mlflowloggedmodel` | The updated LoggedModel. |
+------------+--------------------------+--------------------------+

.. _mlflowGetLoggedModelResponse:

Response
--------






+------------+--------------------------+----------------------------+
| Field Name |           Type           |        Description         |
+============+==========================+============================+
| model      | :ref:`mlflowloggedmodel` | The retrieved LoggedModel. |
+------------+--------------------------+----------------------------+

.. _mlflowSearchLoggedModelsResponse:

Response
--------






+-----------------+--------------------------------------+--------------------------------------------------------------------+
|   Field Name    |                 Type                 |                            Description                             |
+=================+======================================+====================================================================+
| models          | An array of :ref:`mlflowloggedmodel` | Logged Models that match the search criteria.                      |
+-----------------+--------------------------------------+--------------------------------------------------------------------+
| next_page_token | ``STRING``                           | Token that can be used to retrieve the next page of Logged Models. |
+-----------------+--------------------------------------+--------------------------------------------------------------------+

.. _mlflowSetLoggedModelTagsResponse:

Response
--------






+------------+--------------------------+--------------------------+
| Field Name |           Type           |       Description        |
+============+==========================+==========================+
| model      | :ref:`mlflowloggedmodel` | The updated LoggedModel. |
+------------+--------------------------+--------------------------+

.. _mlflowListLoggedModelArtifactsResponse:

Response
--------






+-----------------+-----------------------------------+----------------------------------------------------------------------+
|   Field Name    |               Type                |                             Description                              |
+=================+===================================+======================================================================+
| root_uri        | ``STRING``                        | Root artifact directory for the logged model.                        |
+-----------------+-----------------------------------+----------------------------------------------------------------------+
| files           | An array of :ref:`mlflowfileinfo` | File location and metadata for artifacts.                            |
+-----------------+-----------------------------------+----------------------------------------------------------------------+
| next_page_token | ``STRING``                        | Token that can be used to retrieve the next page of artifact results |
+-----------------+-----------------------------------+----------------------------------------------------------------------+

.. _mlflowSearchTracesV3Response:

Response
--------






+-----------------+--------------------------------------+----------------------------------------------------------+
|   Field Name    |                 Type                 |                       Description                        |
+=================+======================================+==========================================================+
| traces          | An array of :ref:`mlflowtraceinfov3` | Information about traces that match the search criteria. |
+-----------------+--------------------------------------+----------------------------------------------------------+
| next_page_token | ``STRING``                           |                                                          |
+-----------------+--------------------------------------+----------------------------------------------------------+

.. _mlflowCreateDatasetResponse:

Response
--------






+------------+------------------------------+---------------------+
| Field Name |             Type             |     Description     |
+============+==============================+=====================+
| dataset    | :ref:`mlflowdatasetsdataset` | The created dataset |
+------------+------------------------------+---------------------+

.. _mlflowGetDatasetResponse:

Response
--------






+-----------------+------------------------------+------------------------------------------------+
|   Field Name    |             Type             |                  Description                   |
+=================+==============================+================================================+
| dataset         | :ref:`mlflowdatasetsdataset` | The dataset (without records for lazy loading) |
+-----------------+------------------------------+------------------------------------------------+
| next_page_token | ``STRING``                   | Next page token if more records exist          |
+-----------------+------------------------------+------------------------------------------------+

.. _mlflowSearchEvaluationDatasetsResponse:

Response
--------






+-----------------+------------------------------------------+---------------------------------------+
|   Field Name    |                   Type                   |              Description              |
+=================+==========================================+=======================================+
| datasets        | An array of :ref:`mlflowdatasetsdataset` | List of datasets (metadata only)      |
+-----------------+------------------------------------------+---------------------------------------+
| next_page_token | ``STRING``                               | Next page token if more results exist |
+-----------------+------------------------------------------+---------------------------------------+

.. _mlflowSetDatasetTagsResponse:

Response
--------






+------------+------------------------------+---------------------+
| Field Name |             Type             |     Description     |
+============+==============================+=====================+
| dataset    | :ref:`mlflowdatasetsdataset` | The updated dataset |
+------------+------------------------------+---------------------+

.. _mlflowUpsertDatasetRecordsResponse:

Response
--------






+----------------+-----------+----------------------------+
|   Field Name   |   Type    |        Description         |
+================+===========+============================+
| inserted_count | ``INT32`` | Number of records inserted |
+----------------+-----------+----------------------------+
| updated_count  | ``INT32`` | Number of records updated  |
+----------------+-----------+----------------------------+

.. _mlflowGetDatasetExperimentIdsResponse:

Response
--------






+----------------+------------------------+----------------------------------------------------+
|   Field Name   |          Type          |                    Description                     |
+================+========================+====================================================+
| experiment_ids | An array of ``STRING`` | List of experiment IDs associated with the dataset |
+----------------+------------------------+----------------------------------------------------+

.. _mlflowGetDatasetRecordsResponse:

Response
--------






+-----------------+------------+--------------------------------------------------------+
|   Field Name    |    Type    |                      Description                       |
+=================+============+========================================================+
| records         | ``STRING`` | Records in the dataset (JSON serialized list)          |
+-----------------+------------+--------------------------------------------------------+
| next_page_token | ``STRING`` | Pagination token for next page (if more records exist) |
+-----------------+------------+--------------------------------------------------------+

.. _mlflowAddDatasetToExperimentsResponse:

Response
--------






+------------+------------------------------+------------------------------------------------------+
| Field Name |             Type             |                     Description                      |
+============+==============================+======================================================+
| dataset    | :ref:`mlflowdatasetsdataset` | The updated dataset with new experiment associations |
+------------+------------------------------+------------------------------------------------------+

.. _mlflowRemoveDatasetFromExperimentsResponse:

Response
--------






+------------+------------------------------+------------------------------------------------------------+
| Field Name |             Type             |                        Description                         |
+============+==============================+============================================================+
| dataset    | :ref:`mlflowdatasetsdataset` | The updated dataset after removing experiment associations |
+------------+------------------------------+------------------------------------------------------------+

.. _mlflowGetSecretsConfigResponse:

Response
--------






+-------------------+----------+---------------------------------------------------------------------------+
|    Field Name     |   Type   |                                Description                                |
+===================+==========+===========================================================================+
| secrets_available | ``BOOL`` | Whether the server is configured to handle secrets (encryption available) |
+-------------------+----------+---------------------------------------------------------------------------+

.. _mlflowRun:

Run
---



A single run.


+------------+-------------------------+---------------+
| Field Name |          Type           |  Description  |
+============+=========================+===============+
| info       | :ref:`mlflowruninfo`    | Run metadata. |
+------------+-------------------------+---------------+
| data       | :ref:`mlflowrundata`    | Run data.     |
+------------+-------------------------+---------------+
| inputs     | :ref:`mlflowruninputs`  | Run inputs.   |
+------------+-------------------------+---------------+
| outputs    | :ref:`mlflowrunoutputs` | Run outputs.  |
+------------+-------------------------+---------------+

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
| run_name        | ``STRING``             | The name of the run.                                                             |
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

.. _mlflowRunInputs:

RunInputs
---------



Run inputs.


+----------------+---------------------------------------+----------------------------+
|   Field Name   |                 Type                  |        Description         |
+================+=======================================+============================+
| dataset_inputs | An array of :ref:`mlflowdatasetinput` | Dataset inputs to the Run. |
+----------------+---------------------------------------+----------------------------+
| model_inputs   | An array of :ref:`mlflowmodelinput`   | Model inputs to the Run.   |
+----------------+---------------------------------------+----------------------------+

.. _mlflowRunOutputs:

RunOutputs
----------



Outputs of a Run.


+---------------+--------------------------------------+---------------------------+
|  Field Name   |                 Type                 |        Description        |
+===============+======================================+===========================+
| model_outputs | An array of :ref:`mlflowmodeloutput` | Model outputs of the Run. |
+---------------+--------------------------------------+---------------------------+

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

.. _mlflowScorer:

Scorer
------



Scorer entity representing a scorer in the database.


+-------------------+------------+------------------------------------------------------------------------+
|    Field Name     |    Type    |                              Description                               |
+===================+============+========================================================================+
| experiment_id     | ``INT32``  | The experiment ID.                                                     |
+-------------------+------------+------------------------------------------------------------------------+
| scorer_name       | ``STRING`` | The scorer name.                                                       |
+-------------------+------------+------------------------------------------------------------------------+
| scorer_version    | ``INT32``  | The scorer version.                                                    |
+-------------------+------------+------------------------------------------------------------------------+
| serialized_scorer | ``STRING`` | The serialized scorer string.                                          |
+-------------------+------------+------------------------------------------------------------------------+
| creation_time     | ``INT64``  | The creation time of the scorer version (in milliseconds since epoch). |
+-------------------+------------+------------------------------------------------------------------------+
| scorer_id         | ``STRING`` | The unique identifier for the scorer.                                  |
+-------------------+------------+------------------------------------------------------------------------+

.. _mlflowSearchDatasets:

SearchDatasets
--------------






+----------------+------------------------+----------------------------------------+
|   Field Name   |          Type          |              Description               |
+================+========================+========================================+
| experiment_ids | An array of ``STRING`` | List of experiment IDs to search over. |
+----------------+------------------------+----------------------------------------+

.. _mlflowSearchEvaluationDatasets:

SearchEvaluationDatasets
------------------------






+----------------+------------------------+----------------------------------------+
|   Field Name   |          Type          |              Description               |
+================+========================+========================================+
| experiment_ids | An array of ``STRING`` | Associated experiment IDs to filter by |
+----------------+------------------------+----------------------------------------+
| filter_string  | ``STRING``             | Filter string for dataset names        |
+----------------+------------------------+----------------------------------------+
| max_results    | ``INT32``              | Maximum number of results              |
+----------------+------------------------+----------------------------------------+
| order_by       | An array of ``STRING`` | Ordering criteria                      |
+----------------+------------------------+----------------------------------------+
| page_token     | ``STRING``             | Page token for pagination              |
+----------------+------------------------+----------------------------------------+

.. _mlflowSearchLoggedModels:

SearchLoggedModels
------------------






+----------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------+
|   Field Name   |                        Type                        |                                          Description                                          |
+================+====================================================+===============================================================================================+
| experiment_ids | An array of ``STRING``                             | IDs of the Experiments in which to search for Logged Models.                                  |
+----------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------+
| filter         | ``STRING``                                         | A filter expression over Logged Model info and data that allows returning a subset of         |
|                |                                                    | Logged Models. The syntax is a subset of SQL that supports ANDing together binary operations  |
|                |                                                    | Example: ``params.alpha < 0.3 AND metrics.accuracy > 0.9``.                                   |
+----------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------+
| datasets       | An array of :ref:`mlflowsearchloggedmodelsdataset` | List of datasets on which to apply the metrics filter clauses.                                |
|                |                                                    | For example, a filter with `metrics.accuracy > 0.9` and dataset info with name "test_dataset" |
|                |                                                    | means we will return all logged models with accuracy > 0.9 on the test_dataset.               |
|                |                                                    | Metric values from ANY dataset matching the criteria are considered.                          |
|                |                                                    | If no datasets are specified, then metrics across all datasets are considered in the filter.  |
+----------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------+
| max_results    | ``INT32``                                          | Maximum number of Logged Models to return. Max threshold is 50.                               |
+----------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------+
| order_by       | An array of :ref:`mlflowsearchloggedmodelsorderby` | List of columns for ordering the results, with additional fields for sorting criteria.        |
+----------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------+
| page_token     | ``STRING``                                         | Token indicating the page of Logged Models to fetch.                                          |
+----------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------+

.. _mlflowSearchTraces:

SearchTraces
------------






+----------------+------------------------+---------------------------------------------------------------------------------------+
|   Field Name   |          Type          |                                      Description                                      |
+================+========================+=======================================================================================+
| experiment_ids | An array of ``STRING`` | List of experiment IDs to search over.                                                |
+----------------+------------------------+---------------------------------------------------------------------------------------+
| filter         | ``STRING``             | A filter expression over trace attributes and tags that allows returning a subset of  |
|                |                        | traces. The syntax is a subset of SQL that supports ANDing together binary operations |
|                |                        | Example: ``trace.status = 'OK' and trace.timestamp_ms > 1711089570679``.              |
+----------------+------------------------+---------------------------------------------------------------------------------------+
| max_results    | ``INT32``              | Maximum number of traces desired. Max threshold is 500.                               |
+----------------+------------------------+---------------------------------------------------------------------------------------+
| order_by       | An array of ``STRING`` | List of columns for ordering the results, e.g. ``["timestamp_ms DESC"]``.             |
+----------------+------------------------+---------------------------------------------------------------------------------------+
| page_token     | ``STRING``             | Token indicating the page of traces to fetch.                                         |
+----------------+------------------------+---------------------------------------------------------------------------------------+

.. _mlflowSearchTracesV3:

SearchTracesV3
--------------






+-------------+----------------------------------------+---------------------------------------------------------------------------------------+
| Field Name  |                  Type                  |                                      Description                                      |
+=============+========================================+=======================================================================================+
| locations   | An array of :ref:`mlflowtracelocation` | A list of MLflow experiments to search over.                                          |
+-------------+----------------------------------------+---------------------------------------------------------------------------------------+
| filter      | ``STRING``                             | A filter expression over trace attributes and tags that allows returning a subset of  |
|             |                                        | traces. The syntax is a subset of SQL that supports ANDing together binary operations |
|             |                                        | Example: ``trace.status = 'OK' and trace.timestamp_ms > 1711089570679``.              |
+-------------+----------------------------------------+---------------------------------------------------------------------------------------+
| max_results | ``INT32``                              | Maximum number of traces desired. Max threshold is 500.                               |
+-------------+----------------------------------------+---------------------------------------------------------------------------------------+
| order_by    | An array of ``STRING``                 | List of columns for ordering the results, e.g. ``["timestamp_ms DESC"]``.             |
+-------------+----------------------------------------+---------------------------------------------------------------------------------------+
| page_token  | ``STRING``                             | Token indicating the page of traces to fetch.                                         |
+-------------+----------------------------------------+---------------------------------------------------------------------------------------+

.. _mlflowSearchUnifiedTraces:

SearchUnifiedTraces
-------------------






+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------+
|    Field Name    |          Type          |                                                    Description                                                    |
+==================+========================+===================================================================================================================+
| model_id         | ``STRING``             |                                                                                                                   |
|                  |                        | This field is required.                                                                                           |
|                  |                        |                                                                                                                   |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------+
| sql_warehouse_id | ``STRING``             |                                                                                                                   |
|                  |                        | This field is required.                                                                                           |
|                  |                        |                                                                                                                   |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------+
| experiment_ids   | An array of ``STRING`` | TODO: Eventually we want to provide an API that only uses model_id                                                |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------+
| filter           | ``STRING``             | A filter expression over trace attributes and tags that allows returning a subset of                              |
|                  |                        | traces. The syntax is a subset of SQL that supports ANDing together binary operations                             |
|                  |                        | Example: ``trace.status = 'OK' and trace.timestamp_ms > 1711089570679``.                                          |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------+
| max_results      | ``INT32``              | Maximum number of traces desired. Max threshold is 500.                                                           |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------+
| order_by         | An array of ``STRING`` | List of columns for ordering the results, e.g. ``["timestamp_ms DESC"]``.                                         |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------+
| page_token       | ``STRING``             | Token indicating the page of traces to fetch. This is a unified token that encodes both online and offline traces |
|                  |                        | tokens.                                                                                                           |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------+

.. _mlflowCreateGatewaySecretSecretValueEntry:

SecretValueEntry
----------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowUpdateGatewaySecretSecretValueEntry:

SecretValueEntry
----------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowassessmentsExpectationSerializedValue:

SerializedValue
---------------






+----------------------+------------+-------------------------------------------------------------------------------------------------------------------------+
|      Field Name      |    Type    |                                                       Description                                                       |
+======================+============+=========================================================================================================================+
| serialization_format | ``STRING`` | Marks the serialization format for the expectation value. This is a contract specific to the client. The service        |
|                      |            | will not attempt to deserialize the value or validate the format. An example format is "JSON_FORMAT".                   |
+----------------------+------------+-------------------------------------------------------------------------------------------------------------------------+
| value                | ``STRING`` | The value of the expectation-based assessment serialized as a string in the format defined by ``serialization_format``. |
+----------------------+------------+-------------------------------------------------------------------------------------------------------------------------+

.. _mlflowSetDatasetTags:

SetDatasetTags
--------------






+------------+------------+-------------------------------+
| Field Name |    Type    |          Description          |
+============+============+===============================+
| dataset_id | ``STRING`` | Dataset ID to update tags for |
|            |            | This field is required.       |
|            |            |                               |
+------------+------------+-------------------------------+
| tags       | ``STRING`` | Tags to update (JSON string). |
|            |            | This field is required.       |
|            |            |                               |
+------------+------------+-------------------------------+

.. _mlflowSetLoggedModelTags:

SetLoggedModelTags
------------------






+------------+-----------------------------------------+----------------------------------------------+
| Field Name |                  Type                   |                 Description                  |
+============+=========================================+==============================================+
| model_id   | ``STRING``                              | The ID of the LoggedModel to set the tag on. |
|            |                                         | This field is required.                      |
|            |                                         |                                              |
+------------+-----------------------------------------+----------------------------------------------+
| tags       | An array of :ref:`mlflowloggedmodeltag` | The tag key.                                 |
+------------+-----------------------------------------+----------------------------------------------+

.. _mlflowSetTraceTag:

SetTraceTag
-----------






+------------+------------+------------------------------------------------------------------------------------+
| Field Name |    Type    |                                    Description                                     |
+============+============+====================================================================================+
| request_id | ``STRING`` | ID of the trace on which to set a tag.                                             |
+------------+------------+------------------------------------------------------------------------------------+
| key        | ``STRING`` | Name of the tag. Maximum size depends on storage backend.                          |
|            |            | All storage backends are guaranteed to support key values up to 250 bytes in size. |
+------------+------------+------------------------------------------------------------------------------------+
| value      | ``STRING`` | String value of the tag being logged. Maximum size depends on storage backend.     |
|            |            | All storage backends are guaranteed to support key values up to 250 bytes in size. |
+------------+------------+------------------------------------------------------------------------------------+

.. _mlflowSetTraceTagV3:

SetTraceTagV3
-------------






+------------+------------+------------------------------------------------------------------------------------+
| Field Name |    Type    |                                    Description                                     |
+============+============+====================================================================================+
| trace_id   | ``STRING`` | ID of the trace on which to set a tag.                                             |
+------------+------------+------------------------------------------------------------------------------------+
| key        | ``STRING`` | Name of the tag. Maximum size depends on storage backend.                          |
|            |            | All storage backends are guaranteed to support key values up to 250 bytes in size. |
+------------+------------+------------------------------------------------------------------------------------+
| value      | ``STRING`` | String value of the tag being logged. Maximum size depends on storage backend.     |
|            |            | All storage backends are guaranteed to support key values up to 250 bytes in size. |
+------------+------------+------------------------------------------------------------------------------------+

.. _mlflowStartTrace:

StartTrace
----------






+------------------+-----------------------------------------------+-----------------------------------------------------------+
|    Field Name    |                     Type                      |                        Description                        |
+==================+===============================================+===========================================================+
| experiment_id    | ``STRING``                                    | ID of the associated experiment.                          |
+------------------+-----------------------------------------------+-----------------------------------------------------------+
| timestamp_ms     | ``INT64``                                     | Unix timestamp of when the trace started in milliseconds. |
+------------------+-----------------------------------------------+-----------------------------------------------------------+
| request_metadata | An array of :ref:`mlflowtracerequestmetadata` | Metadata about the request that initiated the trace.      |
+------------------+-----------------------------------------------+-----------------------------------------------------------+
| tags             | An array of :ref:`mlflowtracetag`             | Tags for the trace.                                       |
+------------------+-----------------------------------------------+-----------------------------------------------------------+

.. _mlflowStartTraceV3:

StartTraceV3
------------






+------------+--------------------+----------------------------------------------+
| Field Name |        Type        |                 Description                  |
+============+====================+==============================================+
| trace      | :ref:`mlflowtrace` | The information for the trace being created. |
|            |                    | This field is required.                      |
|            |                    |                                              |
+------------+--------------------+----------------------------------------------+

.. _mlflowTraceInfoV3TagsEntry:

TagsEntry
---------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowTrace:

Trace
-----






+------------+--------------------------------------------------+-------------+
| Field Name |                       Type                       | Description |
+============+==================================================+=============+
| trace_info | :ref:`mlflowtraceinfov3`                         |             |
+------------+--------------------------------------------------+-------------+
| spans      | An array of :ref:`opentelemetryprototracev1span` |             |
+------------+--------------------------------------------------+-------------+

.. _mlflowTraceInfo:

TraceInfo
---------



TraceInfo. Represents metadata of a trace.


+-------------------+-----------------------------------------------+-----------------------------------------------------------------+
|    Field Name     |                     Type                      |                           Description                           |
+===================+===============================================+=================================================================+
| request_id        | ``STRING``                                    | Unique identifier for the trace.                                |
+-------------------+-----------------------------------------------+-----------------------------------------------------------------+
| experiment_id     | ``STRING``                                    | The ID of the experiment that contains the trace.               |
+-------------------+-----------------------------------------------+-----------------------------------------------------------------+
| timestamp_ms      | ``INT64``                                     | Unix timestamp of when the trace started in milliseconds.       |
+-------------------+-----------------------------------------------+-----------------------------------------------------------------+
| execution_time_ms | ``INT64``                                     | Unix timestamp of the duration of the trace in milliseconds.    |
+-------------------+-----------------------------------------------+-----------------------------------------------------------------+
| status            | :ref:`mlflowtracestatus`                      | Overall status of the operation being traced (OK, error, etc.). |
+-------------------+-----------------------------------------------+-----------------------------------------------------------------+
| request_metadata  | An array of :ref:`mlflowtracerequestmetadata` | Other trace metadata.                                           |
+-------------------+-----------------------------------------------+-----------------------------------------------------------------+
| tags              | An array of :ref:`mlflowtracetag`             | Tags for the trace.                                             |
+-------------------+-----------------------------------------------+-----------------------------------------------------------------+

.. _mlflowTraceInfoV3:

TraceInfoV3
-----------






+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
|     Field Name     |                          Type                          |                                                      Description                                                       |
+====================+========================================================+========================================================================================================================+
| trace_id           | ``STRING``                                             | The primary key associated with the trace                                                                              |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| client_request_id  | ``STRING``                                             | Client supplied request ID associated with the trace. This could be used to identify the trace/request from an         |
|                    |                                                        | external system that produced the trace.                                                                               |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_location     | :ref:`mlflowtracelocation`                             |                                                                                                                        |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| request            | ``STRING``                                             | [Deprecated, please use `request_preview` instead.]                                                                    |
|                    |                                                        | Request to the model/agent.                                                                                            |
|                    |                                                        | Equivalent to the input of the root span but added for ease of access.                                                 |
|                    |                                                        | Represented as a JSON string.                                                                                          |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| response           | ``STRING``                                             | [Deprecated, please use `request_preview` instead.]                                                                    |
|                    |                                                        | Response of the model/agent.                                                                                           |
|                    |                                                        | Equivalent to the output of the root span but added for ease of access.                                                |
|                    |                                                        | Represented as a JSON string.                                                                                          |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| request_preview    | ``STRING``                                             | A preview of the request to the model/agent represented as a JSON string. This is equivalent to the input of the root  |
|                    |                                                        | span. This preview value is truncated to 10KB while the full request is stored in the trace data in blob storage.      |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| response_preview   | ``STRING``                                             | A preview of the request to the model/agent represented as a JSON string. This is equivalent to the output of the root |
|                    |                                                        | span. This preview value is truncated to 10KB while the full response is stored in the trace data in blob storage.     |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| request_time       | :ref:`googleprotobuftimestamp`                         | Start time of the trace                                                                                                |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| execution_duration | :ref:`googleprotobufduration`                          | Execution time of the trace                                                                                            |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| state              | :ref:`mlflowtraceinfov3state`                          |                                                                                                                        |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_metadata     | An array of :ref:`mlflowtraceinfov3tracemetadataentry` | Metadata associated with the trace.                                                                                    |
|                    |                                                        | Examples include:                                                                                                      |
|                    |                                                        | - run_id: The ID of the mlflow Run (i.e. evaluation job) that produced the trace. May not be                           |
|                    |                                                        |           applicable in certain situations such as if the trace was created via interactive vibe checks)               |
|                    |                                                        | - model_id: The ID of the associated model that produced the trace.                                                    |
|                    |                                                        | - dataset_id: The ID of the mlflow Dataset (usually used together with dataset_record_id)                              |
|                    |                                                        | - dataset_record_id: The ID of the mlflow Dataset (usually used together with dataset_record_id)                       |
|                    |                                                        | - session_id: The ID of the session (e.g. chat conversation) where the request came from                               |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| assessments        | An array of :ref:`mlflowassessmentsassessment`         |                                                                                                                        |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| tags               | An array of :ref:`mlflowtraceinfov3tagsentry`          | Mutable, user-defined tags for the trace, e.g. "question_topic": "DBSQL"                                               |
+--------------------+--------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+

.. _mlflowTraceLocation:

TraceLocation
-------------



The location where the traces was stored and produced


+----------------------------------------------+--------------------------------------------------------------------------------------------------------+----------------------------+
|                  Field Name                  |                                                  Type                                                  |        Description         |
+==============================================+========================================================================================================+============================+
| type                                         | :ref:`mlflowtracelocationtracelocationtype`                                                            |                            |
+----------------------------------------------+--------------------------------------------------------------------------------------------------------+----------------------------+
| ``mlflow_experiment`` OR ``inference_table`` | :ref:`mlflowtracelocationmlflowexperimentlocation` OR :ref:`mlflowtracelocationinferencetablelocation` |                            |
|                                              |                                                                                                        |                            |
|                                              |                                                                                                        | If ``mlflow_experiment``,  |
|                                              |                                                                                                        |                            |
|                                              |                                                                                                        |                            |
|                                              |                                                                                                        |                            |
|                                              |                                                                                                        |                            |
|                                              |                                                                                                        |                            |
|                                              |                                                                                                        | If ``inference_table``,    |
+----------------------------------------------+--------------------------------------------------------------------------------------------------------+----------------------------+

.. _mlflowTraceInfoV3TraceMetadataEntry:

TraceMetadataEntry
------------------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowTraceRequestMetadata:

TraceRequestMetadata
--------------------






+------------+------------+----------------------------------+
| Field Name |    Type    |           Description            |
+============+============+==================================+
| key        | ``STRING`` | Key identifying this metadata.   |
+------------+------------+----------------------------------+
| value      | ``STRING`` | Value identifying this metadata. |
+------------+------------+----------------------------------+

.. _mlflowTraceTag:

TraceTag
--------






+------------+------------+---------------------------------------+
| Field Name |    Type    |              Description              |
+============+============+=======================================+
| key        | ``STRING`` | Key identifying this trace tag.       |
+------------+------------+---------------------------------------+
| value      | ``STRING`` | Value associated with this trace tag. |
+------------+------------+---------------------------------------+

.. _mlflowUpdateAssessment:

UpdateAssessment
----------------



A request to update an existing assessment.


+-------------+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| Field Name  |                Type                |                                                          Description                                                          |
+=============+====================================+===============================================================================================================================+
| assessment  | :ref:`mlflowassessmentsassessment` | The Assessment containing the fields which should be updated.                                                                 |
|             |                                    | This field is required.                                                                                                       |
|             |                                    |                                                                                                                               |
+-------------+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| update_mask | :ref:`googleprotobuffieldmask`     | The list of the assessment fields to update. These should correspond to the values (or lack thereof) present in `assessment`. |
|             |                                    | This field is required.                                                                                                       |
|             |                                    |                                                                                                                               |
+-------------+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

.. _mlflowUpsertDatasetRecords:

UpsertDatasetRecords
--------------------






+------------+------------+-----------------------------------------------------------------+
| Field Name |    Type    |                           Description                           |
+============+============+=================================================================+
| dataset_id | ``STRING`` | Dataset ID to upsert records for                                |
|            |            | This field is required.                                         |
|            |            |                                                                 |
+------------+------------+-----------------------------------------------------------------+
| records    | ``STRING`` | Records to upsert (JSON serialized list of record dictionaries) |
|            |            | This field is required.                                         |
|            |            |                                                                 |
+------------+------------+-----------------------------------------------------------------+
| updated_by | ``STRING`` | User performing the update                                      |
+------------+------------+-----------------------------------------------------------------+

.. _mlflowMetricDataPointValuesEntry:

ValuesEntry
-----------






+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``DOUBLE`` |             |
+------------+------------+-------------+

.. _mlflowWebhook:

Webhook
-------



Webhook entity


+------------------------+---------------------------------------+----------------------------------------------+
|       Field Name       |                 Type                  |                 Description                  |
+========================+=======================================+==============================================+
| webhook_id             | ``STRING``                            | Unique identifier for the webhook            |
+------------------------+---------------------------------------+----------------------------------------------+
| name                   | ``STRING``                            | Name of the webhook                          |
+------------------------+---------------------------------------+----------------------------------------------+
| description            | ``STRING``                            | Optional description for the webhook         |
+------------------------+---------------------------------------+----------------------------------------------+
| url                    | ``STRING``                            | URL to send webhook events to                |
+------------------------+---------------------------------------+----------------------------------------------+
| events                 | An array of :ref:`mlflowwebhookevent` | List of events this webhook is subscribed to |
+------------------------+---------------------------------------+----------------------------------------------+
| status                 | :ref:`mlflowwebhookstatus`            | Current status of the webhook                |
+------------------------+---------------------------------------+----------------------------------------------+
| creation_timestamp     | ``INT64``                             | Timestamp when webhook was created           |
+------------------------+---------------------------------------+----------------------------------------------+
| last_updated_timestamp | ``INT64``                             | Timestamp when webhook was last updated      |
+------------------------+---------------------------------------+----------------------------------------------+

.. _mlflowWebhookEvent:

WebhookEvent
------------



Webhook event definition


+------------+----------------------------+-------------------------+
| Field Name |            Type            |       Description       |
+============+============================+=========================+
| entity     | :ref:`mlflowwebhookentity` | Entity type (required)  |
|            |                            | This field is required. |
|            |                            |                         |
+------------+----------------------------+-------------------------+
| action     | :ref:`mlflowwebhookaction` | Action type (required)  |
|            |                            | This field is required. |
|            |                            |                         |
+------------+----------------------------+-------------------------+

.. _mlflowWebhookTestResult:

WebhookTestResult
-----------------



Test webhook result


+-----------------+------------+----------------------------------------+
|   Field Name    |    Type    |              Description               |
+=================+============+========================================+
| success         | ``BOOL``   | Whether the test succeeded             |
+-----------------+------------+----------------------------------------+
| response_status | ``INT32``  | HTTP response status code if available |
+-----------------+------------+----------------------------------------+
| response_body   | ``STRING`` | Response body if available             |
+-----------------+------------+----------------------------------------+
| error_message   | ``STRING`` | Error message if test failed           |
+-----------------+------------+----------------------------------------+

.. _mlflowAggregationType:

AggregationType
---------------


Aggregation type for metrics.

+------------+---------------------------------------------------------------+
|    Name    |                          Description                          |
+============+===============================================================+
| COUNT      | Count of entities.                                            |
+------------+---------------------------------------------------------------+
| SUM        | Sum of values.                                                |
+------------+---------------------------------------------------------------+
| AVG        | Average of values.                                            |
+------------+---------------------------------------------------------------+
| PERCENTILE | Percentile aggregation (requires percentile_value parameter). |
+------------+---------------------------------------------------------------+
| MIN        | Minimum value.                                                |
+------------+---------------------------------------------------------------+
| MAX        | Maximum value.                                                |
+------------+---------------------------------------------------------------+

.. _mlflowModelVersionDeploymentJobStateDeploymentJobRunState:

DeploymentJobRunState
---------------------




+--------------------------------------+-------------+
|                 Name                 | Description |
+======================================+=============+
| DEPLOYMENT_JOB_RUN_STATE_UNSPECIFIED |             |
+--------------------------------------+-------------+
| NO_VALID_DEPLOYMENT_JOB_FOUND        |             |
+--------------------------------------+-------------+
| RUNNING                              |             |
+--------------------------------------+-------------+
| SUCCEEDED                            |             |
+--------------------------------------+-------------+
| FAILED                               |             |
+--------------------------------------+-------------+
| PENDING                              |             |
+--------------------------------------+-------------+
| APPROVAL                             |             |
+--------------------------------------+-------------+

.. _mlflowFallbackStrategy:

FallbackStrategy
----------------


Fallback strategy for routing (future-proof for additional strategies)

+-------------------------------+----------------------------------------------------------+
|             Name              |                       Description                        |
+===============================+==========================================================+
| FALLBACK_STRATEGY_UNSPECIFIED |                                                          |
+-------------------------------+----------------------------------------------------------+
| SEQUENTIAL                    | Sequential fallback: tries models in the order specified |
+-------------------------------+----------------------------------------------------------+

.. _mlflowGatewayModelLinkageType:

GatewayModelLinkageType
-----------------------


Type of linkage between endpoint and model definition

+--------------------------+-------------------------------------------+
|           Name           |                Description                |
+==========================+===========================================+
| LINKAGE_TYPE_UNSPECIFIED |                                           |
+--------------------------+-------------------------------------------+
| PRIMARY                  | Primary linkage: used for routing traffic |
+--------------------------+-------------------------------------------+
| FALLBACK                 | Fallback linkage: used for failover       |
+--------------------------+-------------------------------------------+

.. _mlflowJobStatus:

JobStatus
---------


Generic status enum for MLflow jobs.
Can be used across different job types (optimization, scorer, etc.).

+------------------------+----------------------------------+
|          Name          |           Description            |
+========================+==================================+
| JOB_STATUS_UNSPECIFIED |                                  |
+------------------------+----------------------------------+
| JOB_STATUS_PENDING     | Job is queued, waiting to start. |
+------------------------+----------------------------------+
| JOB_STATUS_IN_PROGRESS | Job is currently running.        |
+------------------------+----------------------------------+
| JOB_STATUS_COMPLETED   | Job completed successfully.      |
+------------------------+----------------------------------+
| JOB_STATUS_FAILED      | Job failed with an error.        |
+------------------------+----------------------------------+
| JOB_STATUS_CANCELED    | Job was canceled by user.        |
+------------------------+----------------------------------+

.. _mlflowLoggedModelStatus:

LoggedModelStatus
-----------------


A LoggedModelStatus enum value represents the status of a logged
model.

+---------------------------------+--------------------------------------------------------------------------------+
|              Name               |                                  Description                                   |
+=================================+================================================================================+
| LOGGED_MODEL_STATUS_UNSPECIFIED |                                                                                |
+---------------------------------+--------------------------------------------------------------------------------+
| LOGGED_MODEL_PENDING            | The LoggedModel has been created, but the LoggedModel files are not            |
|                                 | completely uploaded.                                                           |
+---------------------------------+--------------------------------------------------------------------------------+
| LOGGED_MODEL_READY              | The LoggedModel is created, and the LoggedModel files are completely uploaded. |
+---------------------------------+--------------------------------------------------------------------------------+
| LOGGED_MODEL_UPLOAD_FAILED      | The LoggedModel is created, but an error occurred when uploading the           |
|                                 | LoggedModel files such as model weights / agent code.                          |
+---------------------------------+--------------------------------------------------------------------------------+

.. _mlflowMetricViewType:

MetricViewType
--------------


View type for metrics aggregation.

+-------------+--------------------------------+
|    Name     |          Description           |
+=============+================================+
| TRACES      | Aggregate at trace level.      |
+-------------+--------------------------------+
| SPANS       | Aggregate at span level.       |
+-------------+--------------------------------+
| ASSESSMENTS | Aggregate at assessment level. |
+-------------+--------------------------------+

.. _mlflowModelVersionStatus:

ModelVersionStatus
------------------




+----------------------+-----------------------------------------------------------------------------------------+
|         Name         |                                       Description                                       |
+======================+=========================================================================================+
| PENDING_REGISTRATION | Request to register a new model version is pending as server performs background tasks. |
+----------------------+-----------------------------------------------------------------------------------------+
| FAILED_REGISTRATION  | Request to register a new model version has failed.                                     |
+----------------------+-----------------------------------------------------------------------------------------+
| READY                | Model version is ready for use.                                                         |
+----------------------+-----------------------------------------------------------------------------------------+

.. _mlflowOptimizerType:

OptimizerType
-------------


Type of optimizer algorithm to use.

+----------------------------+------------------------------------------------------------------------------------------+
|            Name            |                                       Description                                        |
+============================+==========================================================================================+
| OPTIMIZER_TYPE_UNSPECIFIED |                                                                                          |
+----------------------------+------------------------------------------------------------------------------------------+
| OPTIMIZER_TYPE_GEPA        | GEPA (Genetic Pareto) optimizer (https://github.com/gepa-ai/gepa)                        |
+----------------------------+------------------------------------------------------------------------------------------+
| OPTIMIZER_TYPE_METAPROMPT  | MetaPrompt optimizer - uses metaprompting with LLMs to improve prompts in a single pass. |
+----------------------------+------------------------------------------------------------------------------------------+

.. _mlflowRoutingStrategy:

RoutingStrategy
---------------


Routing strategy for endpoints

+------------------------------+-------------------------------------------------------------------+
|             Name             |                            Description                            |
+==============================+===================================================================+
| ROUTING_STRATEGY_UNSPECIFIED |                                                                   |
+------------------------------+-------------------------------------------------------------------+
| REQUEST_BASED_TRAFFIC_SPLIT  | Request-based traffic split: distributes traffic based on weights |
+------------------------------+-------------------------------------------------------------------+

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

.. _mlflowSourceType:

SourceType
----------


Source that generated a run.

+----------+------------------------------------------------------------------------+
|   Name   |                              Description                               |
+==========+========================================================================+
| NOTEBOOK | Databricks notebook environment.                                       |
+----------+------------------------------------------------------------------------+
| JOB      | Scheduled or Run Now job.                                              |
+----------+------------------------------------------------------------------------+
| PROJECT  | As a prepackaged project: either a Docker image or GitHub source, etc. |
+----------+------------------------------------------------------------------------+
| LOCAL    | Local run: Using CLI, IDE, or local notebook.                          |
+----------+------------------------------------------------------------------------+
| UNKNOWN  | Unknown source type.                                                   |
+----------+------------------------------------------------------------------------+

.. _mlflowassessmentsAssessmentSourceSourceType:

SourceType
----------


Type of the assessment source.

+-------------------------+-------------------------------------------+
|          Name           |                Description                |
+=========================+===========================================+
| SOURCE_TYPE_UNSPECIFIED |                                           |
+-------------------------+-------------------------------------------+
| HUMAN                   | Assessment from a human.                  |
+-------------------------+-------------------------------------------+
| LLM_JUDGE               | Assessment from an LLM Judge.             |
+-------------------------+-------------------------------------------+
| CODE                    | Code-based assessment, (e.g. Python UDF). |
+-------------------------+-------------------------------------------+

.. _mlflowdatasetsDatasetRecordSourceSourceType:

SourceType
----------


Type of the dataset record source.

+-------------------------+-------------------------------+
|          Name           |          Description          |
+=========================+===============================+
| SOURCE_TYPE_UNSPECIFIED |                               |
+-------------------------+-------------------------------+
| TRACE                   | Record from a trace/span.     |
+-------------------------+-------------------------------+
| HUMAN                   | Record from human annotation. |
+-------------------------+-------------------------------+
| DOCUMENT                | Record from a document.       |
+-------------------------+-------------------------------+
| CODE                    | Record from code/computation. |
+-------------------------+-------------------------------+

.. _mlflowTraceInfoV3State:

State
-----


Execution state of the trace at the time that it was logged.

+-------------------+----------------------------------------------------------------------------------------------------------------+
|       Name        |                                                  Description                                                   |
+===================+================================================================================================================+
| STATE_UNSPECIFIED |                                                                                                                |
+-------------------+----------------------------------------------------------------------------------------------------------------+
| OK                | The operation being traced was successful.                                                                     |
+-------------------+----------------------------------------------------------------------------------------------------------------+
| ERROR             | The operation being traced failed.                                                                             |
+-------------------+----------------------------------------------------------------------------------------------------------------+
| IN_PROGRESS       | The operation being traced is still in progress. This is useful for incremental/distributed tracing logging in |
|                   | contrast with when the full trace is logged only upon its completion.                                          |
+-------------------+----------------------------------------------------------------------------------------------------------------+

.. _mlflowDeploymentJobConnectionState:

State
-----




+---------------------------------------------+------------------------------------------------------------------------------------+
|                    Name                     |                                    Description                                     |
+=============================================+====================================================================================+
| DEPLOYMENT_JOB_CONNECTION_STATE_UNSPECIFIED |                                                                                    |
+---------------------------------------------+------------------------------------------------------------------------------------+
| NOT_SET_UP                                  | default state                                                                      |
+---------------------------------------------+------------------------------------------------------------------------------------+
| CONNECTED                                   | connected job: job exists, owner has ACLs, and required job parameters are present |
+---------------------------------------------+------------------------------------------------------------------------------------+
| NOT_FOUND                                   | job was deleted OR owner had job ACLs removed                                      |
+---------------------------------------------+------------------------------------------------------------------------------------+
| REQUIRED_PARAMETERS_CHANGED                 | required job parameters were changed                                               |
+---------------------------------------------+------------------------------------------------------------------------------------+

.. _mlflowTraceLocationTraceLocationType:

TraceLocationType
-----------------




+---------------------------------+-------------+
|              Name               | Description |
+=================================+=============+
| TRACE_LOCATION_TYPE_UNSPECIFIED |             |
+---------------------------------+-------------+
| MLFLOW_EXPERIMENT               |             |
+---------------------------------+-------------+
| INFERENCE_TABLE                 |             |
+---------------------------------+-------------+

.. _mlflowTraceStatus:

TraceStatus
-----------




+--------------------------+--------------------------------------------------+
|           Name           |                   Description                    |
+==========================+==================================================+
| TRACE_STATUS_UNSPECIFIED |                                                  |
+--------------------------+--------------------------------------------------+
| OK                       | The operation being traced was successful.       |
+--------------------------+--------------------------------------------------+
| ERROR                    | The operation being traced failed.               |
+--------------------------+--------------------------------------------------+
| IN_PROGRESS              | The operation being traced is still in progress. |
+--------------------------+--------------------------------------------------+

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

.. _mlflowWebhookAction:

WebhookAction
-------------


Webhook action types

+--------------------+-------------+
|        Name        | Description |
+====================+=============+
| ACTION_UNSPECIFIED |             |
+--------------------+-------------+
| CREATED            |             |
+--------------------+-------------+
| UPDATED            |             |
+--------------------+-------------+
| DELETED            |             |
+--------------------+-------------+
| SET                |             |
+--------------------+-------------+

.. _mlflowWebhookEntity:

WebhookEntity
-------------


Webhook entity types

+---------------------+-------------+
|        Name         | Description |
+=====================+=============+
| ENTITY_UNSPECIFIED  |             |
+---------------------+-------------+
| REGISTERED_MODEL    |             |
+---------------------+-------------+
| MODEL_VERSION       |             |
+---------------------+-------------+
| MODEL_VERSION_TAG   |             |
+---------------------+-------------+
| MODEL_VERSION_ALIAS |             |
+---------------------+-------------+
| PROMPT              |             |
+---------------------+-------------+
| PROMPT_VERSION      |             |
+---------------------+-------------+
| PROMPT_TAG          |             |
+---------------------+-------------+
| PROMPT_VERSION_TAG  |             |
+---------------------+-------------+
| PROMPT_ALIAS        |             |
+---------------------+-------------+

.. _mlflowWebhookStatus:

WebhookStatus
-------------


Webhook status enumeration

+----------+-------------+
|   Name   | Description |
+==========+=============+
| ACTIVE   |             |
+----------+-------------+
| DISABLED |             |
+----------+-------------+