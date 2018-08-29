
.. _rest-api:

========
REST API
========


The MLflow REST API allows you to create, list, and get experiments and runs, and log params, metrics, and artifacts.
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
| artifact_location | ``STRING`` | Location where all artifacts for this experiment are stored.           |
|                   |            | If not provided, the remote server will select an appropriate default. |
+-------------------+------------+------------------------------------------------------------------------+

.. _mlflowCreateExperimentResponse:

Response Structure
------------------






+---------------+-----------+-------------------------------------------+
|  Field Name   |   Type    |                Description                |
+===============+===========+===========================================+
| experiment_id | ``INT64`` | Unique identifier for created experiment. |
+---------------+-----------+-------------------------------------------+

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
| source_version   | ``STRING``                      | Git version of the source code used to create run.                                             |
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

Get metadata, params, tags, and metrics for run. Only the last logged value for each metric is
returned.




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

Log a metric for a run (e.g. ML model accuracy). A metric is a key-value pair (string key,
float value) with an associated timestamp. Within a run, a metric may be logged multiple times.




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



.. _mlflowMlflowServicelogParam:

Log Param
=========


+-------------------------------------------+-------------+
|                 Endpoint                  | HTTP Method |
+===========================================+=============+
| ``2.0/preview/mlflow/runs/log-parameter`` | ``POST``    |
+-------------------------------------------+-------------+

Log a param used for this run. Examples are hyperparameters used for ML model training, or
constant dates and values used in an ETL pipeline. A param is a key-value pair (string key,
string value). A param may only be logged once for a given run.




.. _mlflowLogParam:

Request Structure
-----------------






+------------+------------+---------------------------------------------+
| Field Name |    Type    |                 Description                 |
+============+============+=============================================+
| run_uuid   | ``STRING`` | ID of the run under which to log the param. |
|            |            | This field is required.                     |
|            |            |                                             |
+------------+------------+---------------------------------------------+
| key        | ``STRING`` | Name of the param.                          |
|            |            | This field is required.                     |
|            |            |                                             |
+------------+------------+---------------------------------------------+
| value      | ``STRING`` | String value of the param being logged.     |
|            |            | This field is required.                     |
|            |            |                                             |
+------------+------------+---------------------------------------------+

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

Retrieve the logged value for a metric during a run. For a run, if this metric is logged more
than once, this API retrieves only the latest value logged.




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

Returns a list of all values for the specified metric for a given run.




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

List artifacts for a given run. Takes an optional ``artifact_path`` prefix - if specified,
the response will contain only artifacts with the specified prefix.




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



.. _mlflowMlflowServicegetArtifact:

Get Artifact
============


+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/preview/mlflow/artifacts/get`` | ``GET``     |
+--------------------------------------+-------------+

Streams the contents of the specified artifact.




.. _mlflowGetArtifact:

Request Structure
-----------------






+------------+------------+--------------------------------------------------------------------------------------+
| Field Name |    Type    |                                     Description                                      |
+============+============+======================================================================================+
| run_uuid   | ``STRING`` | ID of the run from which to fetch the artifact.                                      |
+------------+------------+--------------------------------------------------------------------------------------+
| path       | ``STRING`` | Path of the artifact to fetch (relative to the root artifact directory for the run). |
+------------+------------+--------------------------------------------------------------------------------------+

===========================



.. _mlflowMlflowServiceupdateRun:

Update Run
==========


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/preview/mlflow/runs/update`` | ``POST``    |
+------------------------------------+-------------+






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


+-------------------+------------+----------------------------------------------------------+
|    Field Name     |    Type    |                       Description                        |
+===================+============+==========================================================+
| experiment_id     | ``INT64``  | Unique identifier for the experiment.                    |
+-------------------+------------+----------------------------------------------------------+
| name              | ``STRING`` | Human readable name that identifies this experiment.     |
+-------------------+------------+----------------------------------------------------------+
| artifact_location | ``STRING`` | Location where artifacts for this experiment are stored. |
+-------------------+------------+----------------------------------------------------------+

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


+------------+---------------------------------+-------------+
| Field Name |              Type               | Description |
+============+=================================+=============+
| metrics    | An array of :ref:`mlflowmetric` |             |
+------------+---------------------------------+-------------+
| params     | An array of :ref:`mlflowparam`  |             |
+------------+---------------------------------+-------------+

.. _mlflowRunInfo:

RunInfo
-------



Metadata of a single run.


+------------------+---------------------------------+----------------------------------------------------------------------------------+
|    Field Name    |              Type               |                                   Description                                    |
+==================+=================================+==================================================================================+
| run_uuid         | ``STRING``                      | Unique identifier for the run.                                                   |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| experiment_id    | ``INT64``                       | The experiment ID.                                                               |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| name             | ``STRING``                      | Human readable name that identifies this run.                                    |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| source_type      | :ref:`mlflowsourcetype`         | Source type.                                                                     |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| source_name      | ``STRING``                      | Source identifier: GitHub URL, name of notebook, name of job, etc.               |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| user_id          | ``STRING``                      | User who initiated the run.                                                      |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| status           | :ref:`mlflowrunstatus`          | Current status of the run.                                                       |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| start_time       | ``INT64``                       | Unix timestamp of when the run started in milliseconds.                          |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| end_time         | ``INT64``                       | Unix timestamp of when the run ended in milliseconds.                            |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| source_version   | ``STRING``                      | Git commit of the code used for the run.                                         |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| entry_point_name | ``STRING``                      | Name of the entry point for the run.                                             |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| tags             | An array of :ref:`mlflowruntag` | Additional metadata key-value pairs.                                             |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| artifact_uri     | ``STRING``                      | URI of the directory where artifacts should be uploaded.                         |
|                  |                                 | This can be a local path (starting with "/"), or a distributed file system (DFS) |
|                  |                                 | path, like ``s3://bucket/directory`` or ``dbfs:/my/directory``.                  |
|                  |                                 | If not set, the local ``./mlruns`` directory is  chosen.                         |
+------------------+---------------------------------+----------------------------------------------------------------------------------+

.. _mlflowRunTag:

RunTag
------



Tag for a run.


+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+

.. _mlflowSearchExpression:

SearchExpression
----------------






+-----------------------------+-------------------------------------------------------------------------------+--------------------+
|         Field Name          |                                     Type                                      |    Description     |
+=============================+===============================================================================+====================+
| ``metric`` OR ``parameter`` | :ref:`mlflowmetricsearchexpression` OR :ref:`mlflowparametersearchexpression` |                    |
|                             |                                                                               |                    |
|                             |                                                                               | If ``metric``,     |
|                             |                                                                               |                    |
|                             |                                                                               |                    |
|                             |                                                                               |                    |
|                             |                                                                               |                    |
|                             |                                                                               |                    |
|                             |                                                                               | If ``parameter``,  |
+-----------------------------+-------------------------------------------------------------------------------+--------------------+

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

+-----------+----------------------------------+
|   Name    |           Description            |
+===========+==================================+
| RUNNING   | Has been initiated               |
+-----------+----------------------------------+
| SCHEDULED | Scheduled to run at a later time |
+-----------+----------------------------------+
| FINISHED  | Run has completed                |
+-----------+----------------------------------+
| FAILED    | Execution failed                 |
+-----------+----------------------------------+
| KILLED    | Run killed by user               |
+-----------+----------------------------------+

.. _mlflowSourceType:

SourceType
----------


Description of the source that generated a run.

+----------+----------------------------------------------------------------------------+
|   Name   |                                Description                                 |
+==========+============================================================================+
| NOTEBOOK | Within Databricks Notebook environment.                                    |
+----------+----------------------------------------------------------------------------+
| JOB      | Scheduled or Run Now Job.                                                  |
+----------+----------------------------------------------------------------------------+
| PROJECT  | As a prepackaged project: either a docker image or github source, ... etc. |
+----------+----------------------------------------------------------------------------+
| LOCAL    | Local run: Using CLI, IDE, or local notebook                               |
+----------+----------------------------------------------------------------------------+
| UNKNOWN  | Unknown source type                                                        |
+----------+----------------------------------------------------------------------------+