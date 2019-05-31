
.. _rest-api:

========
REST API
========


The MLflow REST API allows you to create, list, and get experiments and runs, and log parameters, metrics, and artifacts.
The API is hosted under the ``/api`` route on the MLflow tracking server. For example, to list
experiments on a tracking server hosted at ``http://localhost:5000``, access
``http://localhost:5000/api/2.0/preview/mlflow/experiments/list``.

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
Validates that another experiment with the same name does not already exist and fails if
another experiment with the same name already exists.


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

Get metadata for an experiment and a list of runs for the experiment.
This method works on deleted experiments.




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






+------------+----------------------------------+----------------------------------------------------------------------------+
| Field Name |               Type               |                                Description                                 |
+============+==================================+============================================================================+
| experiment | :ref:`mlflowexperiment`          | Experiment details.                                                        |
+------------+----------------------------------+----------------------------------------------------------------------------+
| runs       | An array of :ref:`mlflowruninfo` | All (max limit to be imposed) active runs associated with this experiment. |
+------------+----------------------------------+----------------------------------------------------------------------------+

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
| key        | ``STRING`` | Name of the tag. Maximum size is 255 bytes.                                                |
|            |            | This field is required.                                                                    |
|            |            |                                                                                            |
+------------+------------+--------------------------------------------------------------------------------------------+
| value      | ``STRING`` | String value of the tag being logged. Maximum size is 5000 bytes.                          |
|            |            | This field is required.                                                                    |
|            |            |                                                                                            |
+------------+------------+--------------------------------------------------------------------------------------------+

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
|                |                        |                                                                                                      |
|                |                        | You cannot provide ``filter`` when ``anded_expressions`` is present; an ``INVALID_PARAMETER_VALUE``  |
|                |                        | error will be returned if both are specified.                                                        |
|                |                        | If both ``filter`` and ``anded_expressions`` are absent, all runs part of the given experiments      |
|                |                        | are returned.                                                                                        |
+----------------+------------------------+------------------------------------------------------------------------------------------------------+
| run_view_type  | :ref:`mlflowviewtype`  | Whether to display only active, only deleted, or all runs.                                           |
|                |                        | Defaults to only active runs.                                                                        |
+----------------+------------------------+------------------------------------------------------------------------------------------------------+
| max_results    | ``INT32``              | Maximum number of runs desired. Max threshold is 50000                                               |
+----------------+------------------------+------------------------------------------------------------------------------------------------------+
| order_by       | An array of ``STRING`` | Ordering expressions like "tags.`model class` DESC"                                                  |
+----------------+------------------------+------------------------------------------------------------------------------------------------------+

.. _mlflowSearchRunsResponse:

Response Structure
------------------






+------------+------------------------------+--------------------------------------+
| Field Name |             Type             |             Description              |
+============+==============================+======================================+
| runs       | An array of :ref:`mlflowrun` | Runs that match the search criteria. |
+------------+------------------------------+--------------------------------------+

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



.. _RESTadd:

Data Structures
===============



.. _mlflowExperiment:

Experiment
----------



Experiment


+-------------------+------------+--------------------------------------------------------------------+
|    Field Name     |    Type    |                            Description                             |
+===================+============+====================================================================+
| experiment_id     | ``STRING`` | Unique identifier for the experiment.                              |
+-------------------+------------+--------------------------------------------------------------------+
| name              | ``STRING`` | Human readable name that identifies the experiment.                |
+-------------------+------------+--------------------------------------------------------------------+
| artifact_location | ``STRING`` | Location where artifacts for the experiment are stored.            |
+-------------------+------------+--------------------------------------------------------------------+
| lifecycle_stage   | ``STRING`` | Current life cycle stage of the experiment: "active" or "deleted". |
|                   |            | Deleted experiments are not returned by APIs.                      |
+-------------------+------------+--------------------------------------------------------------------+
| last_update_time  | ``INT64``  | Last update time                                                   |
+-------------------+------------+--------------------------------------------------------------------+
| creation_time     | ``INT64``  | Creation time                                                      |
+-------------------+------------+--------------------------------------------------------------------+

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