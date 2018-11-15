
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


+-------------------------------------------+-------------+
|                 Endpoint                  | HTTP Method |
+===========================================+=============+
| ``2.0/preview/mlflow/experiments/create`` | ``POST``    |
+-------------------------------------------+-------------+

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






+---------------+-----------+---------------------------------------+
|  Field Name   |   Type    |              Description              |
+===============+===========+=======================================+
| experiment_id | ``INT64`` | Unique identifier for the experiment. |
+---------------+-----------+---------------------------------------+

===========================



.. _mlflowMlflowServicelistExperiments:

List Experiments
================


+-----------------------------------------+-------------+
|                Endpoint                 | HTTP Method |
+=========================================+=============+
| ``2.0/preview/mlflow/experiments/list`` | ``GET``     |
+-----------------------------------------+-------------+

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






+-------------+-------------------------------------+-----------------+
| Field Name  |                Type                 |   Description   |
+=============+=====================================+=================+
| experiments | An array of :ref:`mlflowexperiment` | All experiments |
+-------------+-------------------------------------+-----------------+

===========================



.. _mlflowMlflowServicegetExperiment:

Get Experiment
==============


+----------------------------------------+-------------+
|                Endpoint                | HTTP Method |
+========================================+=============+
| ``2.0/preview/mlflow/experiments/get`` | ``GET``     |
+----------------------------------------+-------------+

Get metadata for an experiment and a list of runs for the experiment.




.. _mlflowGetExperiment:

Request Structure
-----------------






+---------------+-----------+----------------------------------+
|  Field Name   |   Type    |           Description            |
+===============+===========+==================================+
| experiment_id | ``INT64`` | Identifier to get an experiment. |
|               |           | This field is required.          |
|               |           |                                  |
+---------------+-----------+----------------------------------+

.. _mlflowGetExperimentResponse:

Response Structure
------------------






+------------+----------------------------------+---------------------------------------------------------------------+
| Field Name |               Type               |                             Description                             |
+============+==================================+=====================================================================+
| experiment | :ref:`mlflowexperiment`          | Returns experiment details.                                         |
+------------+----------------------------------+---------------------------------------------------------------------+
| runs       | An array of :ref:`mlflowruninfo` | All (max limit to be imposed) runs associated with this experiment. |
+------------+----------------------------------+---------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicedeleteExperiment:

Experiments Delete
=========================


+-------------------------------------------+-------------+
|                 Endpoint                  | HTTP Method |
+===========================================+=============+
| ``2.0/preview/mlflow/experiments/delete`` | ``POST``    |
+-------------------------------------------+-------------+

Mark an experiment and associated runs, params, metrics, ... etc for deletion.
If the experiment uses FileStore, artifacts associated with experiment are also deleted.




.. _mlflowDeleteExperiment:

Request Structure
-----------------






+---------------+-----------+---------------------------------+
|  Field Name   |   Type    |           Description           |
+===============+===========+=================================+
| experiment_id | ``INT64`` | ID of the associated experiment |
|               |           | This field is required.         |
|               |           |                                 |
+---------------+-----------+---------------------------------+

===========================



.. _mlflowMlflowServicerestoreExperiment:

Experiments Restore
==========================


+--------------------------------------------+-------------+
|                  Endpoint                  | HTTP Method |
+============================================+=============+
| ``2.0/preview/mlflow/experiments/restore`` | ``POST``    |
+--------------------------------------------+-------------+

Restore an experiment marked for deletion. This also restores
associated metadata, runs, metrics, and params. If experiment uses FileStore, underlying
artifacts associated with experiment are also restored.

Throws ``RESOURCE_DOES_NOT_EXIST`` if experiment was never created or was permanently deleted.




.. _mlflowRestoreExperiment:

Request Structure
-----------------






+---------------+-----------+---------------------------------+
|  Field Name   |   Type    |           Description           |
+===============+===========+=================================+
| experiment_id | ``INT64`` | ID of the associated experiment |
|               |           | This field is required.         |
|               |           |                                 |
+---------------+-----------+---------------------------------+

===========================



.. _mlflowMlflowServicecreateRun:

Create Run
==========


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/preview/mlflow/runs/create`` | ``POST``    |
+------------------------------------+-------------+

Create a new run within an experiment. A run is usually a single execution of a
machine learning or data ETL pipeline. MLflow uses runs to track :ref:`mlflowParam`,
:ref:`mlflowMetric`, and :ref:`mlflowRunTag` associated with a single execution.




.. _mlflowCreateRun:

Request Structure
-----------------






+------------------+---------------------------------+------------------------------------------------------------------------------------------------+
|    Field Name    |              Type               |                                          Description                                           |
+==================+=================================+================================================================================================+
| experiment_id    | ``INT64``                       | ID of the associated experiment.                                                               |
+------------------+---------------------------------+------------------------------------------------------------------------------------------------+
| user_id          | ``STRING``                      | ID of the user executing the run.                                                              |
+------------------+---------------------------------+------------------------------------------------------------------------------------------------+
| run_name         | ``STRING``                      | Human readable name for the run.                                                               |
+------------------+---------------------------------+------------------------------------------------------------------------------------------------+
| source_type      | :ref:`mlflowsourcetype`         | Originating source for the run.                                                                |
+------------------+---------------------------------+------------------------------------------------------------------------------------------------+
| source_name      | ``STRING``                      | String descriptor for the run's source. For example, name or description of a notebook, or the |
|                  |                                 | URL or path to a project.                                                                      |
+------------------+---------------------------------+------------------------------------------------------------------------------------------------+
| entry_point_name | ``STRING``                      | Name of the project entry point associated with the current run, if any.                       |
+------------------+---------------------------------+------------------------------------------------------------------------------------------------+
| start_time       | ``INT64``                       | Unix timestamp of when the run started in milliseconds.                                        |
+------------------+---------------------------------+------------------------------------------------------------------------------------------------+
| source_version   | ``STRING``                      | Git commit hash of the source code used to create run.                                         |
+------------------+---------------------------------+------------------------------------------------------------------------------------------------+
| tags             | An array of :ref:`mlflowruntag` | Additional metadata for run.                                                                   |
+------------------+---------------------------------+------------------------------------------------------------------------------------------------+

.. _mlflowCreateRunResponse:

Response Structure
------------------






+------------+------------------+------------------------+
| Field Name |       Type       |      Description       |
+============+==================+========================+
| run        | :ref:`mlflowrun` | The newly created run. |
+------------+------------------+------------------------+

===========================



.. _mlflowMlflowServicegetRun:

Get Run
=======


+---------------------------------+-------------+
|            Endpoint             | HTTP Method |
+=================================+=============+
| ``2.0/preview/mlflow/runs/get`` | ``GET``     |
+---------------------------------+-------------+

Get metadata, params, tags, and metrics for a run. Only the last logged value for each metric
is returned.




.. _mlflowGetRun:

Request Structure
-----------------






+------------+------------+-------------------------+
| Field Name |    Type    |       Description       |
+============+============+=========================+
| run_uuid   | ``STRING`` | ID of the run to fetch. |
|            |            | This field is required. |
|            |            |                         |
+------------+------------+-------------------------+

.. _mlflowGetRunResponse:

Response Structure
------------------






+------------+------------------+-----------------------------------------------------------------------+
| Field Name |       Type       |                              Description                              |
+============+==================+=======================================================================+
| run        | :ref:`mlflowrun` | Run metadata (name, start time, etc) and data (metrics, params, etc). |
+------------+------------------+-----------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicelogMetric:

Log Metric
==========


+----------------------------------------+-------------+
|                Endpoint                | HTTP Method |
+========================================+=============+
| ``2.0/preview/mlflow/runs/log-metric`` | ``POST``    |
+----------------------------------------+-------------+

Log a metric for a run. A metric is a key-value pair (string key, float value) with an 
associated timestamp. Examples include the various metrics that represent ML model accuracy. 
A metric can be logged multiple times.




.. _mlflowLogMetric:

Request Structure
-----------------






+------------+------------+---------------------------------------------------------------+
| Field Name |    Type    |                          Description                          |
+============+============+===============================================================+
| run_uuid   | ``STRING`` | ID of the run under which to log the metric.                  |
|            |            | This field is required.                                       |
|            |            |                                                               |
+------------+------------+---------------------------------------------------------------+
| key        | ``STRING`` | Name of the metric.                                           |
|            |            | This field is required.                                       |
|            |            |                                                               |
+------------+------------+---------------------------------------------------------------+
| value      | ``FLOAT``  | Float value of the metric being logged.                       |
|            |            | This field is required.                                       |
|            |            |                                                               |
+------------+------------+---------------------------------------------------------------+
| timestamp  | ``INT64``  | Unix timestamp in milliseconds at the time metric was logged. |
|            |            | This field is required.                                       |
|            |            |                                                               |
+------------+------------+---------------------------------------------------------------+

===========================



.. _mlflowMlflowServicesetTag:

Set Tag
=======


+-------------------------------------+-------------+
|              Endpoint               | HTTP Method |
+=====================================+=============+
| ``2.0/preview/mlflow/runs/set-tag`` | ``POST``    |
+-------------------------------------+-------------+

Set a tag on a run. Tags are run metadata that can be updated during a run and after
a run completes.




.. _mlflowSetTag:

Request Structure
-----------------






+------------+------------+------------------------------------------------------------------+
| Field Name |    Type    |                           Description                            |
+============+============+==================================================================+
| run_uuid   | ``STRING`` | ID of the run under which to set the tag.                        |
|            |            | This field is required.                                          |
|            |            |                                                                  |
+------------+------------+------------------------------------------------------------------+
| key        | ``STRING`` | Name of the tag. Maximum size is 255 bytes.                      |
|            |            | This field is required.                                          |
|            |            |                                                                  |
+------------+------------+------------------------------------------------------------------+
| value      | ``STRING`` | String value of the tag being logged. Maximum size if 500 bytes. |
|            |            | This field is required.                                          |
|            |            |                                                                  |
+------------+------------+------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicelogParam:

Log Param
=========


+-------------------------------------------+-------------+
|                 Endpoint                  | HTTP Method |
+===========================================+=============+
| ``2.0/preview/mlflow/runs/log-parameter`` | ``POST``    |
+-------------------------------------------+-------------+

Log a param used for a run. A param is a key-value pair (string key,
string value). Examples include hyperparameters used for ML model training and
constant dates and values used in an ETL pipeline. A param can be logged only once for a run.




.. _mlflowLogParam:

Request Structure
-----------------






+------------+------------+--------------------------------------------------------------------+
| Field Name |    Type    |                            Description                             |
+============+============+====================================================================+
| run_uuid   | ``STRING`` | ID of the run under which to log the param.                        |
|            |            | This field is required.                                            |
|            |            |                                                                    |
+------------+------------+--------------------------------------------------------------------+
| key        | ``STRING`` | Name of the param. Maximum size is 255 bytes.                      |
|            |            | This field is required.                                            |
|            |            |                                                                    |
+------------+------------+--------------------------------------------------------------------+
| value      | ``STRING`` | String value of the param being logged. Maximum size if 500 bytes. |
|            |            | This field is required.                                            |
|            |            |                                                                    |
+------------+------------+--------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicegetParam:

Get Param
=========


+-----------------------------------+-------------+
|             Endpoint              | HTTP Method |
+===================================+=============+
| ``2.0/preview/mlflow/params/get`` | ``GET``     |
+-----------------------------------+-------------+

Get a param value.




.. _mlflowGetParam:

Request Structure
-----------------






+------------+------------+-------------------------------------------------------+
| Field Name |    Type    |                      Description                      |
+============+============+=======================================================+
| run_uuid   | ``STRING`` | ID of the run from which to retrieve the param value. |
|            |            | This field is required.                               |
|            |            |                                                       |
+------------+------------+-------------------------------------------------------+
| param_name | ``STRING`` | Name of the param.                                    |
|            |            | This field is required.                               |
|            |            |                                                       |
+------------+------------+-------------------------------------------------------+

.. _mlflowGetParamResponse:

Response Structure
------------------






+------------+--------------------+-----------------------+
| Field Name |        Type        |      Description      |
+============+====================+=======================+
| parameter  | :ref:`mlflowparam` | Param key-value pair. |
+------------+--------------------+-----------------------+

===========================



.. _mlflowMlflowServicegetMetric:

Get Metric
==========


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/preview/mlflow/metrics/get`` | ``GET``     |
+------------------------------------+-------------+

Get the value for a metric logged during a run. If the metric is logged more
than once, returns the last logged value.




.. _mlflowGetMetric:

Request Structure
-----------------






+------------+------------+--------------------------------------------------------+
| Field Name |    Type    |                      Description                       |
+============+============+========================================================+
| run_uuid   | ``STRING`` | ID of the run from which to retrieve the metric value. |
|            |            | This field is required.                                |
|            |            |                                                        |
+------------+------------+--------------------------------------------------------+
| metric_key | ``STRING`` | Name of the metric.                                    |
|            |            | This field is required.                                |
|            |            |                                                        |
+------------+------------+--------------------------------------------------------+

.. _mlflowGetMetricResponse:

Response Structure
------------------






+------------+---------------------+------------------------------------------------+
| Field Name |        Type         |                  Description                   |
+============+=====================+================================================+
| metric     | :ref:`mlflowmetric` | Latest reported value of the specified metric. |
+------------+---------------------+------------------------------------------------+

===========================



.. _mlflowMlflowServicegetMetricHistory:

Get Metric History
==================


+--------------------------------------------+-------------+
|                  Endpoint                  | HTTP Method |
+============================================+=============+
| ``2.0/preview/mlflow/metrics/get-history`` | ``GET``     |
+--------------------------------------------+-------------+

Get a list of all values for the specified metric for a given run.




.. _mlflowGetMetricHistory:

Request Structure
-----------------






+------------+------------+--------------------------------------------------+
| Field Name |    Type    |                   Description                    |
+============+============+==================================================+
| run_uuid   | ``STRING`` | ID of the run from which to fetch metric values. |
|            |            | This field is required.                          |
|            |            |                                                  |
+------------+------------+--------------------------------------------------+
| metric_key | ``STRING`` | Name of the metric.                              |
|            |            | This field is required.                          |
|            |            |                                                  |
+------------+------------+--------------------------------------------------+

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


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/preview/mlflow/runs/search`` | ``POST``    |
+------------------------------------+-------------+

Search for runs that satisfy expressions. Search expressions can use :ref:`mlflowMetric` and
:ref:`mlflowParam` keys.




.. _mlflowSearchRuns:

Request Structure
-----------------






+-------------------+-------------------------------------------+--------------------------------------------------------------------+
|    Field Name     |                   Type                    |                            Description                             |
+===================+===========================================+====================================================================+
| experiment_ids    | An array of ``INT64``                     | List of experiment IDs to search over.                             |
+-------------------+-------------------------------------------+--------------------------------------------------------------------+
| anded_expressions | An array of :ref:`mlflowsearchexpression` | Expressions describing runs (AND-ed together when filtering runs). |
+-------------------+-------------------------------------------+--------------------------------------------------------------------+

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


+---------------------------------------+-------------+
|               Endpoint                | HTTP Method |
+=======================================+=============+
| ``2.0/preview/mlflow/artifacts/list`` | ``GET``     |
+---------------------------------------+-------------+

List artifacts for a run. Takes an optional ``artifact_path`` prefix which if specified,
the response contains only artifacts with the specified prefix.




.. _mlflowListArtifacts:

Request Structure
-----------------






+------------+------------+-----------------------------------------------------------------------------------------+
| Field Name |    Type    |                                       Description                                       |
+============+============+=========================================================================================+
| run_uuid   | ``STRING`` | ID of the run whose artifacts to list.                                                  |
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


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/preview/mlflow/runs/update`` | ``POST``    |
+------------------------------------+-------------+

Update run metadata.




.. _mlflowUpdateRun:

Request Structure
-----------------






+------------+------------------------+-------------------------------------------------------+
| Field Name |          Type          |                      Description                      |
+============+========================+=======================================================+
| run_uuid   | ``STRING``             | ID of the run to update.                              |
|            |                        | This field is required.                               |
|            |                        |                                                       |
+------------+------------------------+-------------------------------------------------------+
| status     | :ref:`mlflowrunstatus` | Updated status of the run.                            |
+------------+------------------------+-------------------------------------------------------+
| end_time   | ``INT64``              | Unix timestamp of when the run ended in milliseconds. |
+------------+------------------------+-------------------------------------------------------+

.. _mlflowUpdateRunResponse:

Response Structure
------------------






+------------+----------------------+------------------------------+
| Field Name |         Type         |         Description          |
+============+======================+==============================+
| run_info   | :ref:`mlflowruninfo` | Updated metadata of the run. |
+------------+----------------------+------------------------------+

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
| experiment_id     | ``INT64``  | Unique identifier for the experiment.                              |
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

.. _mlflowFloatClause:

FloatClause
-----------






+------------+------------+------------------------------------------+
| Field Name |    Type    |               Description                |
+============+============+==========================================+
| comparator | ``STRING`` | OneOf (">", ">=", "==", "!=", "<=", "<") |
+------------+------------+------------------------------------------+
| value      | ``FLOAT``  | Float value for comparison.              |
+------------+------------+------------------------------------------+

.. _mlflowMetric:

Metric
------



Metric associated with a run, represented as a key-value pair.


+------------+------------+--------------------------------------------------+
| Field Name |    Type    |                   Description                    |
+============+============+==================================================+
| key        | ``STRING`` | Key identifying this metric.                     |
+------------+------------+--------------------------------------------------+
| value      | ``FLOAT``  | Value associated with this metric.               |
+------------+------------+--------------------------------------------------+
| timestamp  | ``INT64``  | The timestamp at which this metric was recorded. |
+------------+------------+--------------------------------------------------+

.. _mlflowMetricSearchExpression:

MetricSearchExpression
----------------------






+------------+--------------------------+--------------------------------------------+
| Field Name |           Type           |                Description                 |
+============+==========================+============================================+
| ``float``  | :ref:`mlflowfloatclause` |                                            |
|            |                          |                                            |
|            |                          | If ``float``, float clause for comparison. |
+------------+--------------------------+--------------------------------------------+
| key        | ``STRING``               | :ref:`mlflowMetric` key for search.        |
+------------+--------------------------+--------------------------------------------+

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

.. _mlflowParameterSearchExpression:

ParameterSearchExpression
-------------------------






+------------+---------------------------+----------------------------------------------+
| Field Name |           Type            |                 Description                  |
+============+===========================+==============================================+
| ``string`` | :ref:`mlflowstringclause` |                                              |
|            |                           |                                              |
|            |                           | If ``string``, string clause for comparison. |
+------------+---------------------------+----------------------------------------------+
| key        | ``STRING``                | :ref:`mlflowParam` key for search.           |
+------------+---------------------------+----------------------------------------------+

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



Run data (metrics, params, etc).


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


+------------------+-------------------------+----------------------------------------------------------------------------------+
|    Field Name    |          Type           |                                   Description                                    |
+==================+=========================+==================================================================================+
| run_uuid         | ``STRING``              | Unique identifier for the run.                                                   |
+------------------+-------------------------+----------------------------------------------------------------------------------+
| experiment_id    | ``INT64``               | The experiment ID.                                                               |
+------------------+-------------------------+----------------------------------------------------------------------------------+
| name             | ``STRING``              | Human readable name that identifies this run.                                    |
+------------------+-------------------------+----------------------------------------------------------------------------------+
| source_type      | :ref:`mlflowsourcetype` | Source type.                                                                     |
+------------------+-------------------------+----------------------------------------------------------------------------------+
| source_name      | ``STRING``              | Source identifier: GitHub URL, name of notebook, name of job, etc.               |
+------------------+-------------------------+----------------------------------------------------------------------------------+
| user_id          | ``STRING``              | User who initiated the run.                                                      |
+------------------+-------------------------+----------------------------------------------------------------------------------+
| status           | :ref:`mlflowrunstatus`  | Current status of the run.                                                       |
+------------------+-------------------------+----------------------------------------------------------------------------------+
| start_time       | ``INT64``               | Unix timestamp of when the run started in milliseconds.                          |
+------------------+-------------------------+----------------------------------------------------------------------------------+
| end_time         | ``INT64``               | Unix timestamp of when the run ended in milliseconds.                            |
+------------------+-------------------------+----------------------------------------------------------------------------------+
| source_version   | ``STRING``              | Git commit hash of the code used for the run.                                    |
+------------------+-------------------------+----------------------------------------------------------------------------------+
| entry_point_name | ``STRING``              | Name of the entry point for the run.                                             |
+------------------+-------------------------+----------------------------------------------------------------------------------+
| artifact_uri     | ``STRING``              | URI of the directory where artifacts should be uploaded.                         |
|                  |                         | This can be a local path (starting with "/"), or a distributed file system (DFS) |
|                  |                         | path, like ``s3://bucket/directory`` or ``dbfs:/my/directory``.                  |
|                  |                         | If not set, the local ``./mlruns`` directory is  chosen.                         |
+------------------+-------------------------+----------------------------------------------------------------------------------+

.. _mlflowRunTag:

RunTag
------



Tag for a run.


+------------+------------+----------------+
| Field Name |    Type    | Description    |
+============+============+================+
| key        | ``STRING`` | The tag key.   |
+------------+------------+----------------+
| value      | ``STRING`` | The tag value. |
+------------+------------+----------------+

.. _mlflowSearchExpression:

SearchExpression
----------------






+-----------------------------+-------------------------------------------------------------------------------+--------------------------------------------------+
|         Field Name          |                                     Type                                      |                   Description                    |
+=============================+===============================================================================+==================================================+
| ``metric`` OR ``parameter`` | :ref:`mlflowmetricsearchexpression` OR :ref:`mlflowparametersearchexpression` |                                                  |
|                             |                                                                               |                                                  |
|                             |                                                                               | If ``metric``, a metric search expression.       |
|                             |                                                                               |                                                  |
|                             |                                                                               |                                                  |
|                             |                                                                               |                                                  |
|                             |                                                                               |                                                  |
|                             |                                                                               |                                                  |
|                             |                                                                               | If ``parameter``, a parameter search expression. |
+-----------------------------+-------------------------------------------------------------------------------+--------------------------------------------------+

.. _mlflowStringClause:

StringClause
------------






+------------+------------+------------------------------+
| Field Name |    Type    |         Description          |
+============+============+==============================+
| comparator | ``STRING`` | OneOf ("==", "!=", "~")      |
+------------+------------+------------------------------+
| value      | ``STRING`` | String value for comparison. |
+------------+------------+------------------------------+

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
