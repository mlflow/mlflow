
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

.. _RESTadd:

Data Structures
===============



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
| trace_id   | ``STRING`` | ID of the trace from which to delete the tag. |
+------------+------------+-----------------------------------------------+
| key        | ``STRING`` | Name of the tag to delete.                    |
+------------+------------+-----------------------------------------------+

.. _mlflowDeleteTraceTagV3:

DeleteTraceTagV3
----------------






+------------+------------+-----------------------------------------------+
| Field Name |    Type    |                  Description                  |
+============+============+===============================================+
| request_id | ``STRING`` | ID of the trace from which to delete the tag. |
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

.. _mlflowTraceLocationInferenceTableLocation:

InferenceTableLocation
----------------------






+-----------------+------------+--------------------------------------------------------------------+
|   Field Name    |    Type    |                            Description                             |
+=================+============+====================================================================+
| full_table_name | ``STRING`` | Full inference table name in the form of catalog.schema.table_name |
+-----------------+------------+--------------------------------------------------------------------+

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

.. _mlflowSearchDatasets:

SearchDatasets
--------------






+----------------+------------------------+----------------------------------------+
|   Field Name   |          Type          |              Description               |
+================+========================+========================================+
| experiment_ids | An array of ``STRING`` | List of experiment IDs to search over. |
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






+------------+--------------------------+-------------+
| Field Name |           Type           | Description |
+============+==========================+=============+
| trace_info | :ref:`mlflowtraceinfov3` |             |
+------------+--------------------------+-------------+

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
